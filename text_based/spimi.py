import heapq
import json
import os
import re
import struct
import tempfile
from collections import Counter

MAX_MEMORY_BYTES = 64 * 1024 * 1024
MAX_OPEN_RUNS = 16
OUTPUT_FILE = os.path.join("data", "spimi_index")
TERM_OVERHEAD = 64
POSTING_OVERHEAD = 48

UINT32 = struct.Struct(">I")
INT64 = struct.Struct(">q")
POSTING_SIZE = 12  # 8 bytes id + 4 bytes freq

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text):
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def normalize_doc_id(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit() or (value[:1] == "-" and value[1:].isdigit()):
            return int(value)
    raise ValueError(f"id de documento no soportado: {value!r}")


def iter_documents(filename):
    with open(filename, encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
            return
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_exact(file, size):
    data = file.read(size)
    if len(data) != size:
        raise ValueError("archivo binario incompleto")
    return data


def encode_postings(postings):
    ordered = sorted(postings.items())
    data = bytearray()
    data += UINT32.pack(len(ordered))
    for doc_id, frequency in ordered:
        data += INT64.pack(doc_id)
        data += UINT32.pack(frequency)
    return bytes(data)


def decode_postings(data):
    position = 0
    count = UINT32.unpack_from(data, position)[0]
    position += 4

    postings = []
    for _ in range(count):
        doc_id = INT64.unpack_from(data, position)[0]
        position += 8
        frequency = UINT32.unpack_from(data, position)[0]
        position += 4
        postings.append((doc_id, frequency))
    return postings


def write_term_record(file, term, postings):
    term_bytes = term.encode("utf-8")
    postings_bytes = encode_postings(postings)
    file.write(UINT32.pack(len(term_bytes)))
    file.write(term_bytes)
    file.write(postings_bytes)


def read_run(filename):
    with open(filename, "rb") as file:
        while True:
            header = file.read(4)
            if not header:
                return
            if len(header) != 4:
                raise ValueError("bloque temporal incompleto")

            term_length = UINT32.unpack(header)[0]
            term = read_exact(file, term_length)
            count_bytes = read_exact(file, 4)
            count = UINT32.unpack(count_bytes)[0]
            postings_bytes = read_exact(file, count * POSTING_SIZE)
            yield term, count_bytes + postings_bytes


def merge_runs(filenames):
    heap = []
    streams = []

    try:
        for index, filename in enumerate(filenames):
            stream = read_run(filename)
            streams.append(stream)
            try:
                term, postings = next(stream)
            except StopIteration:
                continue
            heapq.heappush(heap, (term, index, postings, stream))

        while heap:
            term, index, postings, stream = heapq.heappop(heap)
            yield term, postings
            try:
                next_term, next_postings = next(stream)
            except StopIteration:
                continue
            heapq.heappush(heap, (next_term, index, next_postings, stream))
    finally:
        for stream in streams:
            stream.close()


def write_merged_run(filenames, output_filename):
    with open(output_filename, "wb") as output:
        for term, postings in merge_runs(filenames):
            output.write(UINT32.pack(len(term)))
            output.write(term)
            output.write(postings)
    for filename in filenames:
        os.remove(filename)


def merge_progressive(filenames, temporary_dir, max_open_runs=MAX_OPEN_RUNS):
    filenames = list(filenames)
    merge_number = 0

    try:
        while len(filenames) > max_open_runs:
            next_filenames = []
            for start in range(0, len(filenames), max_open_runs):
                group = filenames[start : start + max_open_runs]
                if len(group) == 1:
                    next_filenames.append(group[0])
                    continue

                output_filename = os.path.join(
                    temporary_dir, f"merge_{merge_number}.run"
                )
                write_merged_run(group, output_filename)
                next_filenames.append(output_filename)
                merge_number += 1
            filenames = next_filenames

        yield from merge_runs(filenames)
    finally:
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)


class SPIMIIndex:

    def __init__(
        self,
        max_memory=MAX_MEMORY_BYTES,
        output_file=OUTPUT_FILE,
        max_open_runs=MAX_OPEN_RUNS,
    ):
        self.max_memory = max_memory
        self.output_file = output_file
        self.max_open_runs = max_open_runs

        self.index = {}
        self.doc_lengths = {}
        self.doc_count = 0
        self.memory_bytes = 0
        self.block_count = 0
        self.run_paths = []
        self._temporary_dir = None

        self.term_offsets = {}
        self.loaded = False

    def artifact_paths(self):
        return self.output_file + ".bin", self.output_file + ".json"

    def add_posting(self, term, doc_id, frequency):
        if term not in self.index:
            self.index[term] = {}
            self.memory_bytes += TERM_OVERHEAD + len(term.encode("utf-8"))

        postings = self.index[term]
        if doc_id in postings:
            postings[doc_id] += frequency
            return

        postings[doc_id] = frequency
        self.memory_bytes += POSTING_OVERHEAD

    def add_document(self, document):
        if "id" not in document:
            raise ValueError("documento sin campo 'id'")

        doc_id = normalize_doc_id(document["id"])
        text = document.get("lyrics")
        if text is None:
            text = document.get("text", "")
        tokens = tokenize(str(text))

        self.doc_count += 1
        self.doc_lengths[doc_id] = len(tokens)

        for term, frequency in Counter(tokens).items():
            self.add_posting(term, doc_id, frequency)
            if self.memory_bytes >= self.max_memory:
                self.flush_block()

    def flush_block(self):
        if not self.index:
            return

        filename = os.path.join(
            self._temporary_dir, f"block_{self.block_count}.run"
        )
        with open(filename, "wb") as file:
            for term in sorted(self.index):
                write_term_record(file, term, self.index[term])

        self.run_paths.append(filename)
        self.block_count += 1
        self.index = {}
        self.memory_bytes = 0

    def build(self, documents):
        self.index = {}
        self.doc_lengths = {}
        self.doc_count = 0
        self.memory_bytes = 0
        self.block_count = 0
        self.run_paths = []
        self.term_offsets = {}
        self.loaded = False

        base_dir = os.path.dirname(os.path.abspath(self.output_file))
        os.makedirs(base_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="spimi_") as temporary_dir:
            self._temporary_dir = temporary_dir
            try:
                for document in documents:
                    self.add_document(document)
                self.flush_block()
                self.merge_blocks(temporary_dir)
            finally:
                self._temporary_dir = None
                self.run_paths = []

        return self.output_file

    def build_from_json(self, filename):
        return self.build(iter_documents(filename))

    def merge_blocks(self, temporary_dir):
        index_filename, metadata_filename = self.artifact_paths()
        temporary_index = index_filename + ".part"
        temporary_metadata = metadata_filename + ".part"

        term_offsets = {}
        current_term = None
        current_postings = {}

        try:
            with open(temporary_index, "wb") as output:

                def save_current_term():
                    nonlocal current_term, current_postings
                    if current_term is None:
                        return
                    term_offsets[current_term] = output.tell()
                    write_term_record(output, current_term, current_postings)
                    current_term = None
                    current_postings = {}

                for term_bytes, postings_bytes in merge_progressive(
                    self.run_paths,
                    temporary_dir,
                    max_open_runs=self.max_open_runs,
                ):
                    term = term_bytes.decode("utf-8")
                    if term != current_term:
                        save_current_term()
                        current_term = term
                        current_postings = {}

                    for doc_id, frequency in decode_postings(postings_bytes):
                        current_postings[doc_id] = (
                            current_postings.get(doc_id, 0) + frequency
                        )

                save_current_term()

            metadata = {
                "doc_count": self.doc_count,
                "doc_lengths": self.doc_lengths,
                "term_offsets": term_offsets,
            }
            with open(temporary_metadata, "w", encoding="utf-8") as file:
                json.dump(metadata, file, separators=(",", ":"))

            os.replace(temporary_index, index_filename)
            os.replace(temporary_metadata, metadata_filename)
        except Exception:
            for path in (temporary_index, temporary_metadata):
                if os.path.exists(path):
                    os.remove(path)
            raise

        self.term_offsets = term_offsets
        self.loaded = True

    def load(self, filename=None):
        if filename is not None:
            self.output_file = filename

        _, metadata_filename = self.artifact_paths()
        with open(metadata_filename, encoding="utf-8") as file:
            metadata = json.load(file)

        self.doc_count = metadata["doc_count"]
        self.doc_lengths = {
            normalize_doc_id(doc_id): int(length)
            for doc_id, length in metadata["doc_lengths"].items()
        }
        self.term_offsets = metadata["term_offsets"]
        self.loaded = True
        self.index = {}
        return self

    def get_postings(self, term):
        if not self.loaded:
            self.load()

        term = term.casefold()
        if term not in self.term_offsets:
            return []

        index_filename, _ = self.artifact_paths()
        offset = self.term_offsets[term]

        with open(index_filename, "rb") as file:
            file.seek(offset)
            term_length = UINT32.unpack(read_exact(file, 4))[0]
            read_exact(file, term_length)
            count = UINT32.unpack(read_exact(file, 4))[0]
            postings_bytes = read_exact(file, count * POSTING_SIZE)

        return decode_postings(UINT32.pack(count) + postings_bytes)

    def get_doc_length(self, doc_id):
        try:
            return self.doc_lengths.get(normalize_doc_id(doc_id))
        except ValueError:
            return None
