from token_sql import Token
from ast_sql import (AggFun, AndCond, BetweenCond, BoolValue, ColRef,
                     ColumnDec, CompareCond, CreateIndexStmt,
                     CreateTableStmt, DataType, DeleteStmt, DistanceExpr,
                     DropTableStmt, ExplainStmt, FileOrg,
                     FloatValue, IndexKind, InsertStmt, IntValue,
                     JoinClause, Metrica, OrCond, PointValue, PolygonValue,
                     Programa, RelOp, WithinExpr,
                     SelectItem, SelectStmt, SortDir, StrValue,
                     TransactionStmt, UpdateStmt,
                     METRICAS_VECTORIALES, SimilarityScoreExpr,
                     SimilarToExpr, VectorIndex, ImportCsvStmt, MatchExpr)


# En C++ esto era un runtime_error con la posicion pegada al mensaje.
# Aca la fila y la columna quedan como atributos, para poder armar el
# JSON que Monaco necesita sin tener que parsear el texto del error.
class SqlError(Exception):
    def __init__(self, tipo, mensaje, linea, columna):
        self.tipo = tipo
        self.mensaje = mensaje
        self.linea = linea
        self.columna = columna
        super().__init__("{} en linea {}, columna {}: {}".format(
            tipo, linea, columna, mensaje))


class Parser:

    # =============================
    # Metodos de la clase Parser
    # =============================

    def __init__(self, scanner):
        self.scanner = scanner          # De donde se leen los tokens
        self.previous = None            # Token anterior
        self.current = self.scanner.next_token()   # Token actual
        if self.current.type == Token.Type.ERR:
            self.error_lexico()

    def match(self, ttype):
        if self.check(ttype):
            self.advance()
            return True
        return False

    def check(self, ttype):
        if self.is_at_end():
            return False
        return self.current.type == ttype

    def advance(self):
        if not self.is_at_end():
            temp = self.current
            self.current = self.scanner.next_token()
            self.previous = temp

            if self.check(Token.Type.ERR):
                self.error_lexico()
            return True
        return False

    def is_at_end(self):
        return self.current.type == Token.Type.END

    # La posicion sale del token actual, que es el que no encajo con la regla
    def error(self, mensaje):
        raise SqlError("Error sintáctico", mensaje,
                       self.current.linea, self.current.columna)

    def error_semantico(self, mensaje):
        raise SqlError("Error semántico", mensaje,
                       self.current.linea, self.current.columna)

    def error_lexico(self):
        raise SqlError("Error léxico",
                       "caracter no reconocido '" + self.current.text + "'",
                       self.current.linea, self.current.columna)

    # =============================
    # Reglas gramaticales
    # =============================

    def parse_program(self):
        ast = self.parse_p()
        if not self.is_at_end():
            self.error("se esperaba ';' o el fin de la entrada")
        print("Parseo exitoso")
        return ast

    # P ::= Stmt {; Stmt}* [;]
    def parse_p(self):
        p = Programa()

        p.slist.append(self.parse_stmt())
        while self.match(Token.Type.SEMICOL):
            if self.is_at_end():
                break               # ';' final
            p.slist.append(self.parse_stmt())
        return p

    # Stmt ::= CreateTable | CreateIndex | DropTable | Select | Insert
    #        | Delete | Update | Transaction | Explain
    def parse_stmt(self):
        if self.check(Token.Type.CREATE):
            return self.parse_create()
        elif self.check(Token.Type.DROP):
            return self.parse_drop()
        elif self.check(Token.Type.EXPLAIN):
            return self.parse_explain()
        elif self.check(Token.Type.SELECT):
            return self.parse_select()
        elif self.check(Token.Type.INSERT):
            return self.parse_insert()
        elif self.check(Token.Type.IMPORT):
            return self.parse_import_csv()
        elif self.check(Token.Type.DELETE):
            return self.parse_delete()
        elif self.check(Token.Type.UPDATE):
            return self.parse_update()
        elif (
            self.check(Token.Type.BEGIN)
            or self.check(Token.Type.END_KW)
            or self.check(Token.Type.ROLLBACK)
        ):
            return self.parse_transaction()
        else:
            self.error("se esperaba una sentencia")

    def parse_create(self):
        self.match(Token.Type.CREATE)
        if self.check(Token.Type.TABLE):
            return self.parse_create_table()
        elif self.check(Token.Type.INDEX):
            return self.parse_create_index()
        else:
            self.error("se esperaba TABLE o INDEX")

    # CreateTable ::= CREATE TABLE id ( ColDec {, ColDec}* ) USING FileOrg
    def parse_create_table(self):
        self.match(Token.Type.TABLE)
        ct = CreateTableStmt()

        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        ct.tabla = self.previous.text

        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba (")
        ct.columnas.append(self.parse_column_dec())
        while self.match(Token.Type.COMA):
            ct.columnas.append(self.parse_column_dec())
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba )")

        if not self.match(Token.Type.USING):
            self.error("se esperaba USING")
        if self.match(Token.Type.HEAP):
            ct.org = FileOrg.HEAP_ORG
        elif self.match(Token.Type.SEQUENTIAL):
            ct.org = FileOrg.SEQUENTIAL_ORG
        else:
            self.error("se esperaba HEAP o SEQUENTIAL")
        return ct

    # ColDec ::= id Type [PRIMARY KEY]
    def parse_column_dec(self):
        cd = ColumnDec()

        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la columna")
        cd.nombre = self.previous.text

        if self.match(Token.Type.INT):
            cd.tipo = DataType.INT_TYPE
        elif self.match(Token.Type.FLOAT):
            cd.tipo = DataType.FLOAT_TYPE
        elif self.match(Token.Type.BOOL):
            cd.tipo = DataType.BOOL_TYPE
        elif self.match(Token.Type.DATE):
            cd.tipo = DataType.DATE_TYPE
        elif self.match(Token.Type.POINT):
            cd.tipo = DataType.POINT_TYPE
        elif self.match(Token.Type.VARCHAR):
            cd.tipo = DataType.VARCHAR_TYPE
            if not self.match(Token.Type.LPAREN):
                self.error("se esperaba (")
            if not self.match(Token.Type.NUM):
                self.error("se esperaba la longitud del VARCHAR")
            cd.longitud = int(self.previous.text)
            if not self.match(Token.Type.RPAREN):
                self.error("se esperaba )")
        else:
            self.error("se esperaba un tipo de dato")

        if self.match(Token.Type.PRIMARY):
            if not self.match(Token.Type.KEY):
                self.error("se esperaba KEY")
            cd.primary_key = True
        return cd

    # CreateIndex ::= CREATE INDEX ON id ( id ) USING IndexKind [CLUSTERED]
    # IndexKind ::= BTREE | HASH | BITMAP | RTREE | IVF | HNSW
    def parse_create_index(self):
        self.match(Token.Type.INDEX)
        ci = CreateIndexStmt()

        if not self.match(Token.Type.ON):
            self.error("se esperaba ON")
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        ci.tabla = self.previous.text

        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba (")
        if not self.match(Token.Type.ID):
            self.error("se esperaba la columna a indexar")
        ci.columna = self.previous.text
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba )")

        if not self.match(Token.Type.USING):
            self.error("se esperaba USING")
        if self.match(Token.Type.BTREE):
            ci.tipo = IndexKind.BTREE_IDX
        elif self.match(Token.Type.HASH):
            ci.tipo = IndexKind.HASH_IDX
        elif self.match(Token.Type.BITMAP):
            ci.tipo = IndexKind.BITMAP_IDX
        elif self.match(Token.Type.RTREE):
            ci.tipo = IndexKind.RTREE_IDX
        elif self.match(Token.Type.IVF):
            ci.tipo = IndexKind.IVF_IDX
        elif self.match(Token.Type.HNSW):
            ci.tipo = IndexKind.HNSW_IDX
        elif self.match(Token.Type.SPIMI):
            ci.tipo = IndexKind.SPIMI_IDX
        else:
            self.error("se esperaba BTREE, HASH, BITMAP, RTREE, IVF, HNSW o SPIMI")

        if self.match(Token.Type.CLUSTERED):
            ci.clustered = True

        # Un indice hash agrupado no tiene sentido: el hash no preserva el
        # orden, asi que no puede definir el orden fisico de los registros
        if ci.clustered and ci.tipo == IndexKind.HASH_IDX:
            self.error_semantico("un indice HASH no puede ser CLUSTERED")
        # El bitmap guarda mascaras de bits, no registros ordenados: agrupar
        # por el obligaria a copiar cada fila al heap, que ya es donde vive
        if ci.clustered and ci.tipo == IndexKind.BITMAP_IDX:
            self.error_semantico("un indice BITMAP no puede ser CLUSTERED")
        if ci.clustered and ci.tipo in (IndexKind.IVF_IDX, IndexKind.HNSW_IDX, IndexKind.SPIMI_IDX):
            self.error_semantico("un indice IVF o HNSW no puede ser CLUSTERED")
        return ci

    # DropTable ::= DROP TABLE id
    def parse_drop(self):
        self.match(Token.Type.DROP)
        if not self.match(Token.Type.TABLE):
            self.error("se esperaba TABLE despues de DROP")

        d = DropTableStmt()
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        d.tabla = self.previous.text
        return d

    # Explain ::= EXPLAIN [ANALYZE] Stmt
    def parse_explain(self):
        self.match(Token.Type.EXPLAIN)
        analyze = self.match(Token.Type.ANALYZE)

        if self.check(Token.Type.EXPLAIN):
            self.error("EXPLAIN no se puede anidar")
        sentencia = self.parse_stmt()
        return ExplainStmt(sentencia, analyze)

    # Select ::= SELECT SelList FROM id [Join] [Where [UsingVec]] [GroupBy]
    #            [OrderBy] [Limit]
    def parse_select(self):
        self.match(Token.Type.SELECT)
        s = SelectStmt()

        if self.check(Token.Type.STAR):
            self.match(Token.Type.STAR)
            s.select_all = True
            while self.match(Token.Type.COMA):
                s.proyeccion.append(self.parse_select_item())
        else:
            s.proyeccion.append(self.parse_select_item())
            while self.match(Token.Type.COMA):
                s.proyeccion.append(self.parse_select_item())

        if not self.match(Token.Type.FROM):
            self.error("se esperaba FROM")
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        s.tabla = self.previous.text

        if self.check(Token.Type.JOIN):
            s.join = self.parse_join()

        if self.match(Token.Type.WHERE):
            s.condicion = self.parse_cond()
            if self.check(Token.Type.USING):
                self.parse_using_vectorial(s)

        if self.match(Token.Type.GROUP):
            if not self.match(Token.Type.BY):
                self.error("se esperaba BY")
            s.group_by = self.parse_col_ref()

        if self.match(Token.Type.ORDER):
            if not self.match(Token.Type.BY):
                self.error("se esperaba BY")
            s.order_by = self.parse_order_expr()
            if self.match(Token.Type.ASC):
                s.direccion = SortDir.ASC_DIR
            elif self.match(Token.Type.DESC):
                s.direccion = SortDir.DESC_DIR

        if self.match(Token.Type.LIMIT):
            if not self.match(Token.Type.NUM):
                self.error("se esperaba el limite de filas")
            s.limite = int(self.previous.text)
            s.haylimite = True

        self.validar_similitud(s)
        return s

    # UsingVec ::= USING (IVF | HNSW) [WITH METRIC = id]
    # sin WITH METRIC se usa coseno
    def parse_using_vectorial(self, s):
        self.match(Token.Type.USING)
        if self.match(Token.Type.IVF):
            s.indice_vectorial = VectorIndex.IVF
        elif self.match(Token.Type.HNSW):
            s.indice_vectorial = VectorIndex.HNSW
        else:
            self.error("se esperaba IVF o HNSW despues de USING")

        if self.match(Token.Type.WITH):
            if not self.match(Token.Type.METRIC):
                self.error("se esperaba METRIC despues de WITH")
            if not self.match(Token.Type.EQ):
                self.error("se esperaba = despues de METRIC")
            if not (self.match(Token.Type.ID) or self.match(Token.Type.STR)):
                self.error("se esperaba el nombre de la metrica")
            nombre = self.previous.text.lower()
            if nombre not in METRICAS_VECTORIALES:
                self.error_semantico(f"metrica desconocida '{self.previous.text}', "
                                     "se esperaba euclidean, cosine o dot")
            s.metrica_vectorial = METRICAS_VECTORIALES[nombre]

    # reglas que no se ven en la gramatica: un solo SIMILAR_TO, no dentro de un OR,
    # y USING IVF/HNSW o SIMILARITY_SCORE() solo si hay SIMILAR_TO
    def validar_similitud(self, s):
        similares, en_or = [], []
        self._buscar_similar(s.condicion, False, similares, en_or)
        if len(similares) > 1:
            self.error_semantico("solo se permite un SIMILAR_TO por consulta")
        if en_or:
            self.error_semantico("SIMILAR_TO no puede ir dentro de un OR")

        usa_score = (isinstance(s.order_by, SimilarityScoreExpr)
                     or any(isinstance(i.columna, SimilarityScoreExpr)
                            for i in s.proyeccion))
        tiene_match = isinstance(s.condicion, MatchExpr) or (
            isinstance(s.condicion, AndCond)
            and any(isinstance(c, MatchExpr) for c in s.condicion.condiciones)
        )
        if not similares and not tiene_match:
            if s.indice_vectorial != VectorIndex.NINGUNO:
                self.error_semantico("USING IVF/HNSW solo se usa con SIMILAR_TO")
            if usa_score:
                self.error_semantico("SIMILARITY_SCORE() solo se usa con SIMILAR_TO")

    def _buscar_similar(self, cond, dentro_de_or, similares, en_or):
        if isinstance(cond, SimilarToExpr):
            similares.append(cond)
            if dentro_de_or:
                en_or.append(cond)
        elif isinstance(cond, (AndCond, OrCond)):
            for c in cond.condiciones:
                self._buscar_similar(c, dentro_de_or or isinstance(cond, OrCond),
                                     similares, en_or)

    # Item ::= Proj [AS id]
    def parse_select_item(self):
        item = self.parse_proj()
        if self.match(Token.Type.AS):
            if not (self.match(Token.Type.ID) or self.match(Token.Type.SCORE)):
                self.error("se esperaba un nombre despues de AS")
            item.alias = self.previous.text
        return item

    # Proj ::= ColRef | AggFun ( ColRef | * ) | SIMILARITY_SCORE ( )
    def parse_proj(self):
        item = SelectItem()

        if self.check(Token.Type.SIMILARITY_SCORE) or self.check(Token.Type.SCORE):
            item.columna = self.parse_similarity_score()
            return item

        if self.match(Token.Type.COUNT):
            item.agg = AggFun.COUNT_AGG
        elif self.match(Token.Type.SUM):
            item.agg = AggFun.SUM_AGG
        elif self.match(Token.Type.AVG):
            item.agg = AggFun.AVG_AGG
        elif self.match(Token.Type.MIN):
            item.agg = AggFun.MIN_AGG
        elif self.match(Token.Type.MAX):
            item.agg = AggFun.MAX_AGG

        if item.agg != AggFun.NONE_AGG:
            if not self.match(Token.Type.LPAREN):
                self.error("se esperaba (")
            if self.match(Token.Type.STAR):
                # Solo COUNT admite '*' como argumento
                if item.agg != AggFun.COUNT_AGG:
                    self.error_semantico("solo COUNT admite * como argumento")
                item.estrella = True
            else:
                item.columna = self.parse_col_ref()
            if not self.match(Token.Type.RPAREN):
                self.error("se esperaba )")
            return item

        item.columna = self.parse_col_ref()
        return item

    # Join ::= JOIN id ON ColRef = ColRef
    def parse_join(self):
        self.match(Token.Type.JOIN)
        j = JoinClause()

        if not self.match(Token.Type.ID):
            self.error("se esperaba la tabla del JOIN")
        j.tabla = self.previous.text

        if not self.match(Token.Type.ON):
            self.error("se esperaba ON")
        j.izquierda = self.parse_col_ref()
        if not self.match(Token.Type.EQ):
            self.error("se esperaba = en el JOIN")
        j.derecha = self.parse_col_ref()
        return j

    # Insert ::= INSERT INTO id VALUES ( Value {, Value}* )
    def parse_insert(self):
        self.match(Token.Type.INSERT)
        if not self.match(Token.Type.INTO):
            self.error("se esperaba INTO")
        ins = InsertStmt()

        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        ins.tabla = self.previous.text

        if not self.match(Token.Type.VALUES):
            self.error("se esperaba VALUES")
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba (")
        ins.valores.append(self.parse_value())
        while self.match(Token.Type.COMA):
            ins.valores.append(self.parse_value())
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba )")
        return ins

    # IMPORT INTO tabla FROM 'data/archivo.csv'
    def parse_import_csv(self):
        self.match(Token.Type.IMPORT)
        if not self.match(Token.Type.INTO):
            self.error("se esperaba INTO")
        stmt = ImportCsvStmt()
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        stmt.tabla = self.previous.text
        if not self.match(Token.Type.FROM):
            self.error("se esperaba FROM")
        if not self.match(Token.Type.STR):
            self.error("se esperaba la ruta del archivo CSV entre comillas")
        stmt.archivo = self.previous.text
        return stmt

    # Delete ::= DELETE FROM id WHERE Cond
    def parse_delete(self):
        self.match(Token.Type.DELETE)
        if not self.match(Token.Type.FROM):
            self.error("se esperaba FROM")
        d = DeleteStmt()

        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        d.tabla = self.previous.text

        # WHERE obligatorio: evita borrar la tabla completa por descuido
        if not self.match(Token.Type.WHERE):
            self.error("DELETE requiere WHERE")
        d.condicion = self.parse_cond()
        return d

    # Update ::= UPDATE id SET id = Value {, id = Value} WHERE Cond
    def parse_update(self):
        self.match(Token.Type.UPDATE)
        u = UpdateStmt()

        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la tabla")
        u.tabla = self.previous.text

        if not self.match(Token.Type.SET):
            self.error("se esperaba SET")
        u.asignaciones.append(self.parse_update_assignment())
        while self.match(Token.Type.COMA):
            u.asignaciones.append(self.parse_update_assignment())

        if not self.match(Token.Type.WHERE):
            self.error("UPDATE requiere WHERE")
        u.condicion = self.parse_cond()
        return u

    def parse_update_assignment(self):
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de la columna")
        columna = self.previous.text
        if not self.match(Token.Type.EQ):
            self.error("se esperaba = en la asignacion")
        return columna, self.parse_value()

    # Transaction ::= BEGIN TRANSACTION | END TRANSACTION | ROLLBACK
    def parse_transaction(self):
        if self.match(Token.Type.ROLLBACK):
            return TransactionStmt(es_rollback=True)
        if self.match(Token.Type.BEGIN):
            es_begin = True
        elif self.match(Token.Type.END_KW):
            es_begin = False
        else:
            self.error("se esperaba BEGIN o END")
        if not self.match(Token.Type.TRANSACTION):
            self.error("se esperaba TRANSACTION")
        return TransactionStmt(es_begin)

    # Cond ::= AndCond {OR AndCond}*
    def parse_cond(self):
        l = self.parse_and()
        if not self.check(Token.Type.OR):
            return l

        o = OrCond()
        o.condiciones.append(l)
        while self.match(Token.Type.OR):
            o.condiciones.append(self.parse_and())
        return o

    # AndCond ::= Pred {AND Pred}*
    def parse_and(self):
        l = self.parse_pred()
        if not self.check(Token.Type.AND):
            return l

        a = AndCond()
        a.condiciones.append(l)
        while self.match(Token.Type.AND):
            a.condiciones.append(self.parse_pred())
        return a

    # Pred ::= Operando RelOp Value | ColRef BETWEEN Value AND Value | ( Cond )
    #        | DentroDe | SimilarTo
    # Operando ::= ColRef | Distancia
    def parse_pred(self):
        if self.match(Token.Type.LPAREN):
            c = self.parse_cond()
            if not self.match(Token.Type.RPAREN):
                self.error("se esperaba )")
            return c

        if self.check(Token.Type.SIMILAR_TO):
            return self.parse_similar_to()

        if self.check(Token.Type.MATCH):
            return self.parse_match()

        # Predicado espacial de pertenencia: dentro_de(col, POLYGON(...))
        # No lleva operador: ya es verdadero o falso por si mismo.
        if self.check(Token.Type.DENTRO_DE):
            return self.parse_dentro_de()

        # Predicado espacial: distancia(a, b) <op> valor
        if self._es_distancia():
            expr = self.parse_distancia()
            op = self.parse_rel_op()
            return CompareCond(expr, op, self.parse_value())

        col = self.parse_col_ref()

        if self.match(Token.Type.BETWEEN):
            inf = self.parse_value()
            if not self.match(Token.Type.AND):
                self.error("se esperaba AND en BETWEEN")
            sup = self.parse_value()
            return BetweenCond(col, inf, sup)

        op = self.parse_rel_op()
        v = self.parse_value()
        return CompareCond(col, op, v)

    # RelOp ::= = | != | < | <= | > | >=
    def parse_rel_op(self):
        if self.match(Token.Type.EQ):
            return RelOp.EQ_OP
        if self.match(Token.Type.NEQ):
            return RelOp.NEQ_OP
        if self.match(Token.Type.LT):
            return RelOp.LT_OP
        if self.match(Token.Type.LE):
            return RelOp.LE_OP
        if self.match(Token.Type.GT):
            return RelOp.GT_OP
        if self.match(Token.Type.GE):
            return RelOp.GE_OP
        self.error("se esperaba un operador relacional o BETWEEN")

    def _es_distancia(self) -> bool:
        return (self.check(Token.Type.DISTANCIA)
                or self.check(Token.Type.DISTANCIA_EUCLIDIANA)
                or self.check(Token.Type.DISTANCIA_GEODESICA))

    # Distancia ::= (DISTANCIA | DISTANCIA_EUCLIDIANA | DISTANCIA_GEODESICA)
    #               ( Operando , Operando )
    # Operando  ::= Point | ColRef
    #
    # distancia / distancia_euclidiana -> distancia plana, en grados
    # distancia_geodesica              -> Haversine, en metros
    def parse_distancia(self):
        if self.match(Token.Type.DISTANCIA_GEODESICA):
            metrica = Metrica.GEODESICA
        elif (self.match(Token.Type.DISTANCIA)
              or self.match(Token.Type.DISTANCIA_EUCLIDIANA)):
            metrica = Metrica.EUCLIDIANA
        else:
            self.error("se esperaba una funcion de distancia")

        nombre = self.previous.text
        if not self.match(Token.Type.LPAREN):
            self.error(f"se esperaba ( despues de {nombre}")
        izquierda = self.parse_operando_espacial()
        if not self.match(Token.Type.COMA):
            self.error(f"{nombre} espera dos argumentos separados por ,")
        derecha = self.parse_operando_espacial()
        if not self.match(Token.Type.RPAREN):
            self.error(f"se esperaba ) al cerrar {nombre}")
        return DistanceExpr(izquierda, derecha, metrica)

    # DentroDe ::= DENTRO_DE ( ColRef , Polygon )
    def parse_dentro_de(self):
        self.match(Token.Type.DENTRO_DE)
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de dentro_de")
        columna = self.parse_col_ref()
        if not self.match(Token.Type.COMA):
            self.error("dentro_de espera dos argumentos separados por ,")
        if not self.check(Token.Type.POLYGON):
            self.error("el segundo argumento de dentro_de debe ser un POLYGON")
        poligono = self.parse_polygon()
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba ) al cerrar dentro_de")
        return WithinExpr(columna, poligono)

    # Polygon ::= POLYGON ( Point {, Point}* )
    #
    # Hacen falta al menos tres vertices para delimitar un area. El anillo
    # se cierra solo, no hay que repetir el primero al final.
    def parse_polygon(self):
        self.match(Token.Type.POLYGON)
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de POLYGON")
        vertices = [self.parse_point()]
        while self.match(Token.Type.COMA):
            vertices.append(self.parse_point())
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba ) al cerrar POLYGON")
        if len(vertices) < 3:
            self.error(
                f"un POLYGON necesita al menos 3 vertices, se dieron {len(vertices)}"
            )
        return PolygonValue(vertices)

    def parse_operando_espacial(self):
        if self.check(Token.Type.POINT):
            return self.parse_point()
        return self.parse_col_ref()

    # Point ::= POINT ( longitud , latitud )
    #
    # Orden de PostGIS: primero X (longitud), despues Y (latitud).
    # Lima -> POINT(-77.0428, -12.0464)
    def parse_point(self):
        self.match(Token.Type.POINT)
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de POINT")
        x = self.parse_numero()          # longitud
        if not self.match(Token.Type.COMA):
            self.error("POINT espera dos coordenadas separadas por , "
                       "(longitud, latitud)")
        y = self.parse_numero()          # latitud
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba ) al cerrar POINT")
        return PointValue(x, y)

    # Numero ::= [-] num
    def parse_numero(self) -> float:
        negativo = self.match(Token.Type.MINUS)
        if not self.match(Token.Type.NUM):
            self.error("se esperaba un numero")
        valor = float(self.previous.text)
        return -valor if negativo else valor

    # SimilarTo ::= SIMILAR_TO ( str , [k =] num )
    def parse_similar_to(self):
        self.match(Token.Type.SIMILAR_TO)
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de SIMILAR_TO")
        if not self.match(Token.Type.STR):
            self.error("SIMILAR_TO espera la ruta del archivo entre comillas")
        archivo = self.previous.text
        if not self.match(Token.Type.COMA):
            self.error("se esperaba , y el k en SIMILAR_TO")

        if self.check(Token.Type.ID):
            if self.current.text.lower() != "k":
                self.error("se esperaba k=<numero> en SIMILAR_TO")
            self.advance()
            if not self.match(Token.Type.EQ):
                self.error("se esperaba = despues de k")
        if not self.match(Token.Type.NUM):
            self.error("se esperaba el numero de resultados k")
        if "." in self.previous.text or int(self.previous.text) <= 0:
            self.error_semantico("k debe ser un entero mayor que 0")
        k = int(self.previous.text)

        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba ) al cerrar SIMILAR_TO")
        return SimilarToExpr(archivo, k)

    # Score ::= SIMILARITY_SCORE ( )
    def parse_similarity_score(self):
        if not (self.match(Token.Type.SIMILARITY_SCORE) or self.match(Token.Type.SCORE)):
            self.error("se esperaba SCORE()")
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de SCORE")
        if not self.match(Token.Type.RPAREN):
            self.error("SIMILARITY_SCORE() no recibe argumentos")
        return SimilarityScoreExpr()

    def parse_match(self):
        self.match(Token.Type.MATCH)
        if not self.match(Token.Type.LPAREN):
            self.error("se esperaba ( despues de MATCH")
        columna = self.parse_col_ref()
        if not self.match(Token.Type.COMA):
            self.error("MATCH espera columna y consulta")
        if not self.match(Token.Type.STR):
            self.error("MATCH espera la consulta entre comillas")
        expr = MatchExpr(columna, self.previous.text)
        if not self.match(Token.Type.RPAREN):
            self.error("se esperaba ) al cerrar MATCH")
        if not self.match(Token.Type.USING):
            self.error("MATCH requiere USING BM25 o USING TF_IDF")
        if self.match(Token.Type.BM25):
            expr.algoritmo = "bm25"
        elif self.match(Token.Type.TF_IDF):
            expr.algoritmo = "tfidf"
        else:
            self.error("se esperaba BM25 o TF_IDF despues de USING")
        return expr

    # OrderExpr ::= Distancia | SIMILARITY_SCORE ( ) | ColRef
    # un ColRef tambien puede ser el alias de un AS, eso lo resuelve el executor
    def parse_order_expr(self):
        if self._es_distancia():
            return self.parse_distancia()
        if self.check(Token.Type.SIMILARITY_SCORE):
            return self.parse_similarity_score()
        if self.check(Token.Type.SCORE):
            self.match(Token.Type.SCORE)
            if self.check(Token.Type.LPAREN):
                if not self.match(Token.Type.LPAREN):
                    self.error("se esperaba ( despues de SCORE")
                if not self.match(Token.Type.RPAREN):
                    self.error("SCORE() no recibe argumentos")
                return SimilarityScoreExpr()
            return ColRef(columna=self.previous.text)
        return self.parse_col_ref()

    # ColRef ::= id | id . id
    def parse_col_ref(self):
        if not self.match(Token.Type.ID):
            self.error("se esperaba el nombre de una columna")
        primero = self.previous.text

        c = ColRef()
        if self.match(Token.Type.PUNTO):
            if not self.match(Token.Type.ID):
                self.error("se esperaba la columna despues del .")
            c.tabla = primero
            c.columna = self.previous.text
        else:
            c.columna = primero
        return c

    # Value ::= [-] num | str | TRUE | FALSE | Point
    def parse_value(self):
        if self.check(Token.Type.POINT):
            return self.parse_point()

        if self.check(Token.Type.MINUS):
            self.match(Token.Type.MINUS)
            if not self.match(Token.Type.NUM):
                self.error("se esperaba un numero despues del signo -")
            texto = self.previous.text
            if texto.find('.') != -1:
                return FloatValue(-float(texto))
            return IntValue(-int(texto))

        if self.match(Token.Type.NUM):
            texto = self.previous.text
            if texto.find('.') != -1:
                return FloatValue(float(texto))
            return IntValue(int(texto))
        elif self.match(Token.Type.STR):
            return StrValue(self.previous.text)
        elif self.match(Token.Type.TRUE_KW):
            return BoolValue(True)
        elif self.match(Token.Type.FALSE_KW):
            return BoolValue(False)
        else:
            self.error("se esperaba un valor literal")
