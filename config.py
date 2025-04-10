"""
Configuration management for VaarthaAI.
Centralizes all configuration settings to avoid duplication and enable easy updates.
"""
import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables from .env file
load_dotenv()

class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class BaseConfig:
    """Base configuration settings common to all environments"""
    # Application info
    APP_NAME = "VaarthaAI"
    APP_VERSION = "0.1.0"
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/vaartha.db")
    
    # API keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model settings
    DEFAULT_INDUSTRY = "coworking"
    CONFIDENCE_THRESHOLD = 0.85
    SIMILARITY_THRESHOLD = 85
    
    # Path settings
    DATA_DIR = "data"
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
    REGULATIONS_PATH = os.path.join(DATA_DIR, "regulations")
    SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_data")
    
    # Feature flags
    USE_GROQ = bool(GROQ_API_KEY)
    USE_BERT = True
    ENABLE_RAG = True

    # Streamlit settings
    STREAMLIT_SERVER_HEADLESS = True
    STREAMLIT_BROWSER_SERVER_ADDRESS = "localhost"

class DevelopmentConfig(BaseConfig):
    """Development environment specific configuration"""
    DEBUG = True
    LOGGING_LEVEL = "DEBUG"
    
    # Development-specific overrides
    DATABASE_URL = os.getenv("DEV_DATABASE_URL", BaseConfig.DATABASE_URL)

class TestingConfig(BaseConfig):
    """Testing environment specific configuration"""
    TESTING = True
    LOGGING_LEVEL = "DEBUG"
    
    # Use in-memory database for testing
    DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")
    
    # Disable external API calls during tests
    USE_GROQ = False

class ProductionConfig(BaseConfig):
    """Production environment specific configuration"""
    DEBUG = False
    LOGGING_LEVEL = "INFO"
    
    # Production requires stronger security
    STREAMLIT_SERVER_HEADLESS = True
    
    # Production might use different database URL
    DATABASE_URL = os.getenv("PROD_DATABASE_URL", BaseConfig.DATABASE_URL)

# Get the current application environment
def get_environment():
    """
    Determine the current environment based on environment variables.
    Defaults to development if not specified.
    """
    env = os.getenv("VAARTHA_ENV", "development").lower()
    if env in [e.value for e in Environment]:
        return Environment(env)
    return Environment.DEVELOPMENT

# Get the configuration for the current environment
def get_config():
    """
    Returns the appropriate configuration object for the current environment.
    """
    env = get_environment()
    configs = {
        Environment.DEVELOPMENT: DevelopmentConfig,
        Environment.TESTING: TestingConfig, 
        Environment.PRODUCTION: ProductionConfig
    }
    return configs[env]

# Create a config instance for easy importing
config = get_config()