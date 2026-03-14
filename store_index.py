from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY

extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()
""""
# Puisque le test direct a marché, on utilise cette méthode
api_key = "pcsk_7JSrth_9K4bAQovDV73bXJiqnBKLhJYw6LA4RauVtPzNv311i3AVrpKDM4TPxgN8JVth3Q"
pc = Pinecone(api_key=api_key)

index_name = "medicalbot"

# On vérifie si l'index existe déjà pour éviter de planter
if index_name not in pc.list_indexes().names():
    print(f"Création de l'index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,    # Dimension pour sentence-transformers
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("✅ Index créé avec succès !")
else:
    print(f"ℹ️ L'index '{index_name}' existe déjà.")

"""
# Initialiser le client Pinecone avec ta clé API
pc = Pinecone(api_key=PINECONE_API_KEY)

# Nom de ton index
index_name = "medicalbot"

# Créer l’index avec les spécifications
pc.create_index(
    name=index_name,
    dimension=384,          # taille des vecteurs (dépend du modèle d’embeddings)
    metric="cosine",        # mesure de similarité
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
