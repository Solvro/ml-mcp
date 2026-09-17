"""Refuse generated ingestion Cypher that does more than MERGE the page's own entities.

Issue #93: the extraction model writes Cypher from text crawled out of web documents, and that
Cypher runs as the pipeline's Neo4j user. Every rewrite before it recognises the shapes it
expects and passes anything else through verbatim, so ``MATCH (n) DETACH DELETE n`` or
``LOAD CSV FROM 'http://...'`` reached the database untouched.

The check is positive. The prompt allows MERGE node and relationship statements with properties
set on the page's own variables, and a statement is kept only when it parses as exactly that. A
keyword denylist has to name every clause that can write or reach the network and misses the
next one; here anything the grammar does not describe is refused without anyone having thought
of it.

Retrieval can scan for keywords because the database refuses writes in READ mode behind it.
Ingestion has to write, so nothing stands behind this check, and the statement is tokenised left
to right instead: a quote, a backtick and a comment are each read where they start, the way
Neo4j reads them, so text inside a literal can never hide a clause and a clause can never pass
for a literal.
"""

import re
from dataclasses import dataclass

from src.config.system_labels import SYSTEM_LABELS
from src.text_normalization import CYPHER_STRING_LITERAL_RE

# The first alternative that matches at a position wins, so `//` inside a string is part of the
# string and a quote inside a backticked name is part of the name. Anything nothing matches - `$`,
# `;`, an unterminated quote, a typographic dash Neo4j would read as `-` - refuses the statement
# instead of being skipped.
TOKEN_RE = re.compile(
    r"(?P<space>\s+)"
    r"|(?P<comment>//|/\*)"
    r"|(?P<string>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    r"|(?P<quoted>`[^`]+`)"
    r"|(?P<number>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_]\w*)"
    r"|(?P<operator><>|<=|>=|[()\[\]{}:,.=<>+\-*/%])",
    re.DOTALL,
)

# Pure functions a property value can reasonably be built from. The canonical-key rewrite uses
# `size` and `coalesce`; the rest cover dates and conversions a page may carry. Nothing here reads
# the graph, touches a file or opens a connection. Cypher function names ignore case.
ALLOWED_FUNCTIONS = frozenset(
    {
        "coalesce",
        "date",
        "datetime",
        "duration",
        "left",
        "localdatetime",
        "localtime",
        "ltrim",
        "replace",
        "right",
        "rtrim",
        "size",
        "split",
        "substring",
        "time",
        "toboolean",
        "tofloat",
        "tointeger",
        "tolower",
        "tostring",
        "toupper",
        "trim",
    }
)

# Neo4j accepts most keywords as variable names. Refusing them unquoted keeps this parser and
# Neo4j from ever disagreeing about whether a word is a clause or a variable.
RESERVED_WORDS = frozenset(
    {
        "AND",
        "AS",
        "BY",
        "CALL",
        "CASE",
        "COLLECT",
        "CONTAINS",
        "COUNT",
        "CREATE",
        "CSV",
        "DELETE",
        "DETACH",
        "DROP",
        "ELSE",
        "END",
        "ENDS",
        "EXISTS",
        "FALSE",
        "FOREACH",
        "IN",
        "IS",
        "LIMIT",
        "LOAD",
        "MATCH",
        "MERGE",
        "NOT",
        "NULL",
        "ON",
        "OPTIONAL",
        "OR",
        "ORDER",
        "REMOVE",
        "RETURN",
        "SET",
        "SKIP",
        "STARTS",
        "THEN",
        "TRUE",
        "UNION",
        "UNWIND",
        "WHEN",
        "WHERE",
        "WITH",
        "XOR",
        "YIELD",
    }
)

# Compared without case: `processeddocument` is not the bookkeeping label, but nothing a page
# describes needs to come that close to it.
FOLDED_SYSTEM_LABELS = frozenset(label.lower() for label in SYSTEM_LABELS)

# Every name a refused statement mentions, used only to tell whether a later statement depends on
# it. Over-reading here costs a failed page, never an unsafe statement run.
NAME_RE = re.compile(r"`(?P<quoted>[^`]+)`|(?P<plain>[A-Za-z_]\w*)")


class UnsafeIngestionStatementError(ValueError):
    """Raised when a generated ingestion statement is not a MERGE of the page's own entities."""

    def __init__(self, reason: str, unbound_variable: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.unbound_variable = unbound_variable


class StrandedStatementError(UnsafeIngestionStatementError):
    """Raised when a statement needs a variable that only a refused statement bound."""


@dataclass(frozen=True)
class RefusedStatement:
    """A statement kept out of the page's query, and why."""

    statement: str
    reason: str


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str

    def is_keyword(self, *words: str) -> bool:
        return self.kind == "name" and self.text.upper() in words

    def is_operator(self, *symbols: str) -> bool:
        return self.kind == "operator" and self.text in symbols

    def describe(self) -> str:
        return "the end of the statement" if self.kind == "end" else f"`{self.text}`"


_END = _Token("end", "")


def _tokenize(statement: str) -> list[_Token]:
    """Split a statement into tokens, refusing anything the grammar has no token for."""
    tokens: list[_Token] = []
    position = 0
    while position < len(statement):
        match = TOKEN_RE.match(statement, position)
        if match is None:
            character = statement[position]
            if character in "'\"`":
                raise UnsafeIngestionStatementError(f"unterminated {character} at {position}")
            raise UnsafeIngestionStatementError(f"unexpected character {character!r}")
        kind = match.lastgroup or ""
        if kind == "comment":
            raise UnsafeIngestionStatementError("comments are not allowed")
        if kind != "space":
            tokens.append(_Token(kind, match.group(0)))
        position = match.end()
    return tokens


class _StatementParser:
    """Recursive descent over the one statement shape ingestion may run."""

    def __init__(self, statement: str, bound: frozenset[str]) -> None:
        self._tokens = _tokenize(statement)
        self._position = 0
        self._scope = set(bound)

    def parse(self) -> frozenset[str]:
        if not self._peek().is_keyword("MERGE"):
            raise UnsafeIngestionStatementError(
                f"a statement must start with MERGE, not {self._peek().describe()}"
            )
        while self._peek().kind != "end":
            token = self._next()
            if token.is_keyword("MERGE"):
                self._pattern()
                while self._peek().is_keyword("ON"):
                    self._next()
                    if not self._next().is_keyword("CREATE", "MATCH"):
                        raise UnsafeIngestionStatementError(
                            "ON must be followed by CREATE or MATCH"
                        )
                    self._expect_keyword("SET")
                    self._assignments()
            elif token.is_keyword("SET"):
                self._assignments()
            else:
                raise UnsafeIngestionStatementError(
                    f"{token.describe()} is not allowed: a statement may only MERGE nodes and "
                    "relationships and SET properties on them"
                )
        return frozenset(self._scope)

    def _peek(self) -> _Token:
        return self._at(self._position)

    def _at(self, index: int) -> _Token:
        return self._tokens[index] if index < len(self._tokens) else _END

    def _next(self) -> _Token:
        token = self._peek()
        if token.kind != "end":
            self._position += 1
        return token

    def _expect_operator(self, symbol: str) -> None:
        token = self._next()
        if not token.is_operator(symbol):
            raise UnsafeIngestionStatementError(f"expected `{symbol}`, found {token.describe()}")

    def _expect_keyword(self, word: str) -> None:
        token = self._next()
        if not token.is_keyword(word):
            raise UnsafeIngestionStatementError(f"expected {word}, found {token.describe()}")

    def _variable(self) -> str:
        token = self._next()
        if token.kind == "quoted":
            return token.text[1:-1]
        if token.kind == "name" and token.text.upper() not in RESERVED_WORDS:
            return token.text
        raise UnsafeIngestionStatementError(f"{token.describe()} cannot be used as a variable")

    def _symbolic_name(self, what: str) -> str:
        token = self._next()
        if token.kind == "quoted":
            return token.text[1:-1]
        if token.kind == "name":
            return token.text
        raise UnsafeIngestionStatementError(f"expected {what}, found {token.describe()}")

    def _pattern(self) -> None:
        self._node()
        while self._peek().is_operator("-", "<"):
            self._relationship()
            self._node()

    def _node(self) -> None:
        self._expect_operator("(")
        variable = None
        if self._peek().kind in ("name", "quoted"):
            variable = self._variable()

        labels: list[str] = []
        while self._peek().is_operator(":"):
            self._next()
            labels.append(self._symbolic_name("a label"))
        for label in labels:
            if label.lower() in FOLDED_SYSTEM_LABELS:
                raise UnsafeIngestionStatementError(
                    f"label `{label}` belongs to the pipeline's own bookkeeping"
                )

        has_properties = False
        if self._peek().is_operator("{"):
            has_properties = self._map()
        self._expect_operator(")")

        if variable is not None and variable in self._scope:
            # A node an earlier clause bound. Labels or properties on it would be a redeclaration,
            # which Neo4j refuses before running anything, so that is left for Neo4j to report.
            return
        if variable is not None and not labels and not has_properties:
            raise UnsafeIngestionStatementError(
                f"`({variable})` is not bound by an earlier statement on this page, so it would "
                "match any node in the graph",
                unbound_variable=variable,
            )
        if not labels or not has_properties:
            # Without a label the properties can match a bookkeeping node (`{hash: ...}`), and
            # without properties the label matches every node under it.
            raise UnsafeIngestionStatementError(
                "a new node needs both a label and properties, or it matches nodes the page "
                "did not describe"
            )
        if variable is not None:
            self._scope.add(variable)

    def _relationship(self) -> None:
        if self._peek().is_operator("<"):
            self._next()
        self._expect_operator("-")
        self._expect_operator("[")
        variable = None
        if self._peek().kind in ("name", "quoted"):
            variable = self._variable()
        self._expect_operator(":")
        self._symbolic_name("a relationship type")
        if self._peek().is_operator("{"):
            self._map()
        self._expect_operator("]")
        self._expect_operator("-")
        if self._peek().is_operator(">"):
            self._next()
        if variable is not None:
            self._scope.add(variable)

    def _map(self) -> bool:
        """Parse a property map, returning whether it holds any entry."""
        self._expect_operator("{")
        if self._peek().is_operator("}"):
            self._next()
            return False
        while True:
            self._symbolic_name("a property name")
            self._expect_operator(":")
            self._expression()
            if not self._peek().is_operator(","):
                break
            self._next()
        self._expect_operator("}")
        return True

    def _assignments(self) -> None:
        while True:
            variable = self._variable()
            if variable not in self._scope:
                raise UnsafeIngestionStatementError(
                    f"SET on `{variable}`, which no statement on this page binds",
                    unbound_variable=variable,
                )
            if not self._peek().is_operator("."):
                raise UnsafeIngestionStatementError(
                    f"SET may only assign one property of `{variable}` at a time, not labels "
                    "or a whole map"
                )
            self._next()
            self._symbolic_name("a property name")
            self._expect_operator("=")
            self._expression()
            if not self._peek().is_operator(","):
                return
            self._next()

    def _expression(self) -> None:
        self._conjunction()
        while self._peek().is_keyword("OR", "XOR"):
            self._next()
            self._conjunction()

    def _conjunction(self) -> None:
        self._negation()
        while self._peek().is_keyword("AND"):
            self._next()
            self._negation()

    def _negation(self) -> None:
        if self._peek().is_keyword("NOT"):
            self._next()
            self._negation()
            return
        self._comparison()

    def _comparison(self) -> None:
        self._additive()
        while True:
            token = self._peek()
            if token.is_operator("=", "<>", "<", ">", "<=", ">=") or token.is_keyword(
                "CONTAINS", "IN"
            ):
                self._next()
                self._additive()
            elif token.is_keyword("STARTS", "ENDS"):
                self._next()
                self._expect_keyword("WITH")
                self._additive()
            elif token.is_keyword("IS"):
                self._next()
                if self._peek().is_keyword("NOT"):
                    self._next()
                self._expect_keyword("NULL")
            else:
                return

    def _additive(self) -> None:
        self._multiplicative()
        while self._peek().is_operator("+", "-"):
            self._next()
            self._multiplicative()

    def _multiplicative(self) -> None:
        self._unary()
        while self._peek().is_operator("*", "/", "%"):
            self._next()
            self._unary()

    def _unary(self) -> None:
        if self._peek().is_operator("-"):
            self._next()
            # Only a number. `-(m)` after `(n) -` is how a relationship pattern would read as
            # arithmetic, and a pattern in a value reads the graph.
            if self._next().kind != "number":
                raise UnsafeIngestionStatementError("a minus sign may only negate a number")
            return
        self._atom()

    def _atom(self) -> None:
        token = self._peek()
        if token.kind in ("string", "number") or token.is_keyword("TRUE", "FALSE", "NULL"):
            self._next()
            return
        if token.is_operator("["):
            self._next()
            if not self._peek().is_operator("]"):
                self._expression()
                while self._peek().is_operator(","):
                    self._next()
                    self._expression()
            self._expect_operator("]")
            return
        if token.is_operator("{"):
            self._map()
            return
        if token.is_operator("("):
            self._next()
            self._expression()
            self._expect_operator(")")
            return
        if token.is_keyword("CASE"):
            self._case()
            return
        if token.kind == "name":
            function = self._function_name_ahead()
            if function is not None:
                self._function_call(function)
                return
        if token.kind in ("name", "quoted"):
            self._property_read(self._variable())
            return
        raise UnsafeIngestionStatementError(f"{token.describe()} cannot start a value")

    def _function_name_ahead(self) -> str | None:
        """Return the dotted name of a function call starting here, without consuming it."""
        index = self._position
        parts = [self._at(index).text]
        index += 1
        while self._at(index).is_operator(".") and self._at(index + 1).kind == "name":
            parts.append(self._at(index + 1).text)
            index += 2
        if not self._at(index).is_operator("("):
            return None
        return ".".join(parts)

    def _function_call(self, function: str) -> None:
        if function.lower() not in ALLOWED_FUNCTIONS:
            raise UnsafeIngestionStatementError(f"function `{function}` is not allowed")
        self._next()
        self._expect_operator("(")
        if not self._peek().is_operator(")"):
            self._expression()
            while self._peek().is_operator(","):
                self._next()
                self._expression()
        self._expect_operator(")")

    def _property_read(self, variable: str) -> None:
        if variable not in self._scope:
            raise UnsafeIngestionStatementError(
                f"`{variable}` is not bound by this page", unbound_variable=variable
            )
        if not self._peek().is_operator("."):
            raise UnsafeIngestionStatementError(
                f"only a property of `{variable}` can be read, not the node itself"
            )
        self._next()
        self._symbolic_name("a property name")

    def _case(self) -> None:
        self._expect_keyword("CASE")
        if not self._peek().is_keyword("WHEN"):
            self._expression()
        if not self._peek().is_keyword("WHEN"):
            raise UnsafeIngestionStatementError("CASE needs at least one WHEN")
        while self._peek().is_keyword("WHEN"):
            self._next()
            self._expression()
            self._expect_keyword("THEN")
            self._expression()
        if self._peek().is_keyword("ELSE"):
            self._next()
            self._expression()
        self._expect_keyword("END")


def validate_ingestion_statement(statement: str, bound: frozenset[str]) -> frozenset[str]:
    """
    Check that one generated statement only MERGEs the page's own entities.

    Accepted: ``MERGE`` of a pattern whose new nodes each carry a label and properties and whose
    other nodes are variables an earlier statement bound, followed by ``ON CREATE SET``,
    ``ON MATCH SET`` or ``SET`` assignments of one property at a time on bound variables. Values
    are literals, lists, maps, properties of bound variables, ``CASE`` and ``ALLOWED_FUNCTIONS``.
    No bookkeeping label may appear. Everything else is refused.

    Args:
        statement: One statement from the page, as it would run
        bound: Variables the page's earlier accepted statements bound

    Returns:
        The variables in scope after this statement

    Raises:
        UnsafeIngestionStatementError: If the statement is outside that shape. When the problem
            is a variable nothing bound, ``unbound_variable`` names it.
    """
    try:
        return _StatementParser(statement, bound).parse()
    except RecursionError as error:
        raise UnsafeIngestionStatementError("statement is nested too deeply") from error


def _names_in(statement: str) -> set[str]:
    """Every identifier a statement mentions outside its string literals."""
    without_literals = CYPHER_STRING_LITERAL_RE.sub(" ", statement)
    return {
        match.group("quoted") or match.group("plain")
        for match in NAME_RE.finditer(without_literals)
    }


def refuse_unsafe_statements(statements: list[str]) -> tuple[list[str], list[RefusedStatement]]:
    """
    Keep the statements that only MERGE the page's own entities, in order.

    A refused statement is dropped and the page keeps going - unless a later statement uses a
    variable that only the refused one bound. The page runs as one query, so that later
    statement would either fail it or, as a bare ``(n)``, match any node in the graph; the page
    fails instead, which is what any bad statement did before this check existed.

    Args:
        statements: The page's generated statements

    Returns:
        The statements to run, and the ones refused with the reason for each

    Raises:
        StrandedStatementError: If a statement depends on a variable only a refused one bound
    """
    bound: frozenset[str] = frozenset()
    names_of_refused: set[str] = set()
    kept: list[str] = []
    refused: list[RefusedStatement] = []

    for statement in statements:
        try:
            bound = validate_ingestion_statement(statement, bound)
        except UnsafeIngestionStatementError as error:
            variable = error.unbound_variable
            if variable is not None and variable in names_of_refused:
                raise StrandedStatementError(
                    f"`{variable}` is bound only by a refused statement, and this one uses it: "
                    f"{statement}",
                    unbound_variable=variable,
                ) from error
            refused.append(RefusedStatement(statement, error.reason))
            names_of_refused |= _names_in(statement)
            continue
        kept.append(statement)

    return kept, refused
