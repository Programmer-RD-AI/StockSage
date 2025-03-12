import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def verify_langsmith_setup():
    """Verify LangSmith setup and create project if needed."""

    # Check for required environment variables
    required_vars = ["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"⚠️  Missing environment variables: {', '.join(missing)}")
        print("Please update your .env file with the required LangSmith variables.")
        return False

    # Initialize client and verify connection
    try:
        client = Client()
        client.create_project_if_not_exists(os.getenv("LANGCHAIN_PROJECT"))
        print(f"✅ Successfully connected to LangSmith")
        print(f"✅ Project '{os.getenv('LANGCHAIN_PROJECT')}' is ready")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to LangSmith: {str(e)}")
        return False


if __name__ == "__main__":
    verify_langsmith_setup()
