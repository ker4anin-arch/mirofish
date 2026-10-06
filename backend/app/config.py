"""\n配置管理\n统一从项目根目录的 .env 文件加载配置\n"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Knowledge graph memory (Graphiti + Neo4j, replaces Zep Cloud)
    # NEO4J_HOST alone is enough when the host is injected by the platform
    # (Render's fromService); NEO4J_URI wins when both are set.
    NEO4J_URI = os.environ.get('NEO4J_URI') or (
        f"bolt://{os.environ['NEO4J_HOST']}:7687" if os.environ.get('NEO4J_HOST')
        else 'bolt://localhost:7687'
    )
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
    NEO4J_DATABASE = os.environ.get('NEO4J_DATABASE', 'neo4j')

    # LLM used for graph extraction (defaults to LLM_MODEL_NAME). DeepSeek and
    # most OpenAI-compatible providers need 'json_object'; use 'json_schema'
    # only for providers with native structured outputs.
    GRAPH_LLM_MODEL_NAME = os.environ.get('GRAPH_LLM_MODEL_NAME')
    GRAPH_LLM_MAX_TOKENS = int(os.environ.get('GRAPH_LLM_MAX_TOKENS', '8192'))
    GRAPH_LLM_STRUCTURED_OUTPUT = os.environ.get('GRAPH_LLM_STRUCTURED_OUTPUT', 'json_object')
    GRAPH_MAX_COROUTINES = int(os.environ.get('GRAPH_MAX_COROUTINES', '8'))
    GRAPH_INGESTION_TIMEOUT_SECONDS = int(os.environ.get('GRAPH_INGESTION_TIMEOUT_SECONDS', '7200'))

    # Embeddings: any OpenAI-compatible endpoint, or a local model when
    # EMBEDDING_API_KEY is empty (DeepSeek has no embeddings API).
    EMBEDDING_API_KEY = os.environ.get('EMBEDDING_API_KEY')
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'https://api.openai.com/v1')
    EMBEDDING_MODEL_NAME = os.environ.get(
        'EMBEDDING_MODEL_NAME',
        'text-embedding-3-small' if os.environ.get('EMBEDDING_API_KEY')
        else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    )
    EMBEDDING_DIM = int(os.environ.get('EMBEDDING_DIM', '1024'))
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD 未配置")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
