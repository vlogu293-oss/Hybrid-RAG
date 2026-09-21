from neo4j import GraphDatabase

from .config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


class Neo4jKG:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(
                NEO4J_USERNAME,
                NEO4J_PASSWORD
            )
        )

        self.database = NEO4J_DATABASE

        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def clear_database(self):

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        self.driver.execute_query(
            query,
            database_=self.database
        )

    def create_indexes(self):

        queries = [

            """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:Entity)
            ON (e.name)
            """,

            """
            CREATE INDEX document_id_index IF NOT EXISTS
            FOR (d:Document)
            ON (d.id)
            """
        ]

        for query in queries:

            self.driver.execute_query(
                query,
                database_=self.database
            )

    def add_document(
        self,
        document_id: str,
        text: str
    ):

        query = """
        MERGE (d:Document {id: $document_id})
        SET d.text = $text
        """

        self.driver.execute_query(
            query,
            document_id=document_id,
            text=text,
            database_=self.database
        )

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        document_id: str
    ):

        query = """
        MATCH (d:Document {id: $document_id})

        MERGE (e:Entity {
            name: $entity_name
        })

        SET e.type = $entity_type

        MERGE (d)-[:MENTIONS]->(e)
        """

        self.driver.execute_query(
            query,
            entity_name=entity_name,
            entity_type=entity_type,
            document_id=document_id,
            database_=self.database
        )

    def add_relationship(
        self,
        source: str,
        relation: str,
        target: str
    ):

        safe_relation = relation.upper().replace(
            " ",
            "_"
        )

        query = f"""
        MATCH (a:Entity)
        WHERE toLower(a.name) = toLower($source)

        MATCH (b:Entity)
        WHERE toLower(b.name) = toLower($target)

        MERGE (a)-[r:`{safe_relation}`]->(b)
        """

        self.driver.execute_query(
            query,
            source=source,
            target=target,
            database_=self.database
        )

    def search_entities(
        self,
        entity_names: list[str],
        limit: int = 5
    ):

        if not entity_names:
            return []

        query = """
        MATCH (e:Entity)

        WHERE any(
            name IN $names
            WHERE toLower(e.name) CONTAINS toLower(name)
        )

        OPTIONAL MATCH path =
            (e)-[*1..2]-(related)

        RETURN
            e.name AS entity,
            e.type AS entity_type,
            collect(
                DISTINCT {
                    name: related.name,
                    type: labels(related)
                }
            )[0..$limit] AS related
        LIMIT $limit
        """

        records, _, _ = self.driver.execute_query(
            query,
            names=entity_names,
            limit=limit,
            database_=self.database
        )

        return [
            record.data()
            for record in records
        ]