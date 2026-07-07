import os

from azure.cosmos import CosmosClient
from dotenv import load_dotenv


load_dotenv()

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")


class CosmosReader:

    def __init__(self):
        missing = [
            name
            for name, value in {
                "COSMOS_ENDPOINT": COSMOS_ENDPOINT,
                "COSMOS_KEY": COSMOS_KEY,
                "COSMOS_DATABASE": COSMOS_DATABASE,
                "COSMOS_CONTAINER": COSMOS_CONTAINER,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.client = CosmosClient(
            COSMOS_ENDPOINT,
            COSMOS_KEY
        )

        self.database = self.client.get_database_client(
            COSMOS_DATABASE
        )

        self.container = self.database.get_container_client(
            COSMOS_CONTAINER
        )

    def get_organization_ids(self):
        """
        Returns all unique organization IDs.
        """

        query = """
        SELECT DISTINCT VALUE c.organization_id
        FROM c
        """

        return list(
            self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            )
        )

    def load_documents_by_organization(self, organization_id):
        """
        Streams every document belonging to one organization.
        """

        query = """
        SELECT *
        FROM c
        WHERE c.organization_id = @organizationId
        """

        parameters = [
            {
                "name": "@organizationId",
                "value": organization_id
            }
        ]

        return self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
    def load_document_by_id(self, organization_id, document_id):
        """
        Fetches a single document by its id, scoped to an organization.
        Returns the document dict, or None if not found.
        """

        query = """
        SELECT *
        FROM c
        WHERE c.organization_id = @organizationId
        AND c.id = @documentId
        """

        parameters = [
            {
                "name": "@organizationId",
                "value": organization_id
            },
            {
                "name": "@documentId",
                "value": document_id
            }
        ]

        results = list(
            self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            )
        )

        return results[0] if results else None
    


if __name__ == "__main__":
    reader = CosmosReader()

    print(reader.get_organization_ids())