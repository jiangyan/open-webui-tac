import weaviate
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import weaviate.classes as wvc
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery
import logging
import json
import uuid

from open_webui.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import WEAVIATE_URL

log = logging.getLogger(__name__)

class WeaviateClient:
    def __init__(self):
        self.collection_prefix = "open_webui"
        # Parse URL to get host and port
        parsed_url = urlparse(WEAVIATE_URL)
        
        try:
            self.client = weaviate.connect_to_local(
                host=parsed_url.hostname or "localhost",
                port=int(parsed_url.port or 28080)
            )
        except Exception as e:
            log.error(f"Error connecting to Weaviate: {e}")
            raise

    def __del__(self):
        """Ensure client is closed when object is destroyed"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass

    def _get_class_name(self, collection_name: str) -> str:
        # Weaviate class names must be capitalized and can only contain alphanumeric chars
        sanitized_name = "".join(c for c in collection_name if c.isalnum())
        return f"{self.collection_prefix}_{sanitized_name}".capitalize()

    def _create_collection_if_not_exists(self, collection_name: str):
        class_name = self._get_class_name(collection_name)
        if not self.client.collections.exists(class_name):
            # Create collection with proper V4 configuration
            self.client.collections.create(
                name=class_name,
                properties=[
                    wvc.config.Property(
                        name="text",
                        data_type=wvc.config.DataType.TEXT,
                        tokenization=wvc.config.Tokenization.GSE
                    ),
                    wvc.config.Property(
                        name="metadata",
                        data_type=wvc.config.DataType.OBJECT,
                        nested_properties=[
                            wvc.config.Property(
                                name="source",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="source_id",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="url",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="created_at",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="chunk_id",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="chunk_index",
                                data_type=wvc.config.DataType.INT
                            ),
                            wvc.config.Property(
                                name="embedding_config",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="file_id",
                                data_type=wvc.config.DataType.TEXT
                            ),
                            wvc.config.Property(
                                name="page",
                                data_type=wvc.config.DataType.INT
                            )
                        ]
                    )
                ],
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    ef_construction=128,
                    max_connections=64,
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
                vectorizer_config=None  # We provide vectors directly
            )

    def has_collection(self, collection_name: str) -> bool:
        return self.client.collections.exists(self._get_class_name(collection_name))

    def delete_collection(self, collection_name: str):
        if self.has_collection(collection_name):
            self.client.collections.delete(self._get_class_name(collection_name))

    def search(
        self, collection_name: str, vectors: List[List[float | int]], limit: int, query_str: Optional[str] = None
    ) -> Optional[SearchResult]:
        try:
            class_name = self._get_class_name(collection_name)
            if not self.has_collection(collection_name):
                return None

            collection = self.client.collections.get(class_name)
            
            all_ids = []
            all_distances = []
            all_documents = []
            all_metadatas = []

            for vector in vectors:
                # Use hybrid search if query string is provided
                if query_str:
                    query_result = (
                        collection.query
                        .hybrid(
                            query=query_str,
                            vector=vector,
                            alpha=0.5,  # Balance between vector and keyword search
                            limit=limit,
                            query_properties=["text"],
                            return_metadata=MetadataQuery(score=True, explain_score=True),
                        )
                    )
                else:
                    # Fallback to pure vector search
                    query_result = (
                        collection.query
                        .near_vector(
                            near_vector=vector,
                            limit=limit
                        )
                    )

                if not query_result.objects:
                    continue

                ids = []
                distances = []
                documents = []
                metadatas = []

                for obj in query_result.objects:
                    # Convert UUID to string
                    ids.append(str(obj.uuid))
                    # For hybrid search, use the hybrid score, otherwise use distance
                    if query_str:
                        distances.append(obj.metadata.score)
                    else:
                        distances.append(obj.metadata.get("distance", 1.0))
                    documents.append(obj.properties.get("text"))
                    metadatas.append(self._convert_metadata(obj.properties.get("metadata")))

                all_ids.append(ids)
                all_distances.append(distances)
                all_documents.append(documents)
                all_metadatas.append(metadatas)

            if not all_ids:
                return None

            return SearchResult(
                ids=all_ids,
                distances=all_distances,
                documents=all_documents,
                metadatas=all_metadatas
            )
        except Exception as e:
            log.error(f"Error during search: {e}")
            return None

    def get(self, collection_name: str) -> Optional[GetResult]:
        try:
            class_name = self._get_class_name(collection_name)
            if not self.has_collection(collection_name):
                return None

            collection = self.client.collections.get(class_name)
            # Use V4 query syntax - get all objects with their properties
            query_result = collection.query.fetch_objects()

            if not query_result.objects:
                return None

            ids = []
            documents = []
            metadatas = []

            for obj in query_result.objects:
                # Convert UUID to string
                ids.append(str(obj.uuid))
                documents.append(obj.properties.get("text"))
                metadatas.append(self._convert_metadata(obj.properties.get("metadata")))

            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas]
            )
        except Exception as e:
            log.error(f"Error during get: {e}")
            return None

    def insert(self, collection_name: str, items: List[VectorItem]):
        try:
            self._create_collection_if_not_exists(collection_name)
            collection = self.client.collections.get(self._get_class_name(collection_name))

            with collection.batch.fixed_size(batch_size=100) as batch:
                for item in items:
                    # Ensure metadata has required fields
                    metadata = item["metadata"]
                    if "embedding_config" not in metadata:
                        metadata["embedding_config"] = json.dumps({
                            "engine": metadata.get("embedding_engine", ""),
                            "model": metadata.get("embedding_model", ""),
                            "chunk_size": metadata.get("chunk_size", 0),
                            "chunk_overlap": metadata.get("chunk_overlap", 0)
                        })

                    batch.add_object(
                        properties={
                            "text": item["text"],
                            "metadata": metadata
                        },
                        uuid=item["id"],
                        vector=item["vector"]
                    )
        except Exception as e:
            log.error(f"Error during insert: {e}")
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]):
        try:
            self._create_collection_if_not_exists(collection_name)
            collection = self.client.collections.get(self._get_class_name(collection_name))

            # Batch upsert similar to Chroma's approach
            with collection.batch.fixed_size(batch_size=100) as batch:
                for item in items:
                    # Delete if exists (upsert = delete + insert in Weaviate)
                    try:
                        collection.objects.delete(uuid=item["id"])
                    except:
                        pass
                    # Insert new
                    batch.add_object(
                        properties={
                            "text": item["text"],
                            "metadata": item["metadata"]
                        },
                        uuid=item["id"],
                        vector=item["vector"]
                    )
        except Exception as e:
            print(f"Error during upsert: {e}")
            raise

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ):
        try:
            class_name = self._get_class_name(collection_name)
            if not self.has_collection(collection_name):
                return

            collection = self.client.collections.get(class_name)

            if ids:
                # Delete by IDs similar to Chroma's approach
                for id in ids:
                    try:
                        collection.objects.delete(uuid=id)
                    except:
                        pass
            elif filter:
                # Delete by filter similar to Chroma's approach
                collection.objects.delete_many(
                    where=weaviate.query.Filter.by_property("metadata").equal(filter)
                )
        except Exception as e:
            print(f"Error during delete: {e}")
            raise

    def reset(self):
        try:
            # Delete all collections with our prefix, similar to Chroma's reset
            collections = self.client.collections.list_all()
            for collection in collections:
                if collection.name.startswith(self.collection_prefix):
                    self.client.collections.delete(collection.name)
        except Exception as e:
            print(f"Error during reset: {e}")
            raise 

    def query(
        self,
        collection_name: str,
        filter: Dict[str, Any],
        limit: Optional[int] = None
    ) -> Optional[GetResult]:
        try:
            class_name = self._get_class_name(collection_name)
            if not self.has_collection(collection_name):
                return None

            collection = self.client.collections.get(class_name)
            
            # Build filter for metadata fields
            filter_obj = None
            for key, value in filter.items():
                new_filter = weaviate.query.Filter.by_property(f"metadata.{key}").equal(value)
                if filter_obj is None:
                    filter_obj = new_filter
                else:
                    filter_obj = filter_obj & new_filter

            query_result = (
                collection.query
                .with_where(filter_obj)
                .with_fields("text metadata")
                .with_limit(limit if limit else None)
                .do()
            )

            if not query_result.objects:
                return None

            ids = []
            documents = []
            metadatas = []

            for obj in query_result.objects:
                ids.append(obj.uuid)
                documents.append(obj.properties.get("text"))
                metadatas.append(self._convert_metadata(obj.properties.get("metadata")))

            return GetResult(
                ids=[ids],
                documents=[documents],
                metadatas=[metadatas]
            )
        except Exception as e:
            print(f"Error during query: {e}")
            return None 

    def _convert_metadata(self, metadata: dict) -> dict:
        """Convert any UUID objects in metadata to strings"""
        if not metadata:
            return metadata
        
        converted = {}
        for key, value in metadata.items():
            if isinstance(value, uuid.UUID):
                converted[key] = str(value)
            else:
                converted[key] = value
        return converted 