"""
Secure Configuration Management for Space Analytics Demo

Supports multiple token storage methods with fallback hierarchy.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import getpass

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecureConfigManager:
    """
    Secure configuration manager with multiple token source support.
    
    Token resolution hierarchy:
    1. Environment variables
    2. System keychain (if available)
    3. AWS Secrets Manager (if configured)
    4. Encrypted config file
    5. .env file (fallback)
    6. Interactive prompt (last resort)
    """
    
    def __init__(self, service_name: str = "spaceneo-analytics"):
        self.service_name = service_name
        self.config_cache = {}
        
    def get_token(self, token_name: str, required: bool = False) -> Optional[str]:
        """
        Get token using secure hierarchy.
        
        Args:
            token_name: Name of the token (e.g., 'MOTHERDUCK_TOKEN')
            required: Whether to prompt if token not found
            
        Returns:
            Token value or None if not found and not required
        """
        # Check cache first
        if token_name in self.config_cache:
            return self.config_cache[token_name]
        
        token = None
        source = None
        
        # 1. Environment variables
        token = self._get_from_env(token_name)
        if token:
            source = "environment"
        
        # 2. System keychain
        if not token and KEYRING_AVAILABLE:
            token = self._get_from_keychain(token_name)
            if token:
                source = "keychain"
        
        # 3. AWS Secrets Manager
        if not token and AWS_AVAILABLE:
            token = self._get_from_aws_secrets(token_name)
            if token:
                source = "aws_secrets"
        
        # 4. Encrypted config file
        if not token:
            token = self._get_from_encrypted_config(token_name)
            if token:
                source = "encrypted_config"
        
        # 5. .env file fallback
        if not token:
            token = self._get_from_env_file(token_name)
            if token:
                source = "env_file"
        
        # 6. Interactive prompt (if required)
        if not token and required:
            token = self._get_from_prompt(token_name)
            if token:
                source = "interactive"
                # Store in keychain for future use
                self._store_in_keychain(token_name, token)
        
        if token:
            logger.info("Token '%s' loaded from %s", token_name, source)
            self.config_cache[token_name] = token
        elif required:
            raise ValueError(f"Required token '{token_name}' not found")
        
        return token
    
    def _get_from_env(self, token_name: str) -> Optional[str]:
        """Get token from environment variables."""
        return os.getenv(token_name)
    
    def _get_from_keychain(self, token_name: str) -> Optional[str]:
        """Get token from system keychain."""
        try:
            return keyring.get_password(self.service_name, token_name)
        except Exception as e:
            logger.debug("Keychain access failed: %s", e)
            return None
    
    def _get_from_aws_secrets(self, token_name: str) -> Optional[str]:
        """Get token from AWS Secrets Manager."""
        try:
            secret_name = os.getenv('AWS_SECRET_NAME', f'{self.service_name}/tokens')
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_name)
            secrets = json.loads(response['SecretString'])
            return secrets.get(token_name)
        except Exception as e:
            logger.debug("AWS Secrets Manager access failed: %s", e)
            return None
    
    def _get_from_encrypted_config(self, token_name: str) -> Optional[str]:
        """Get token from encrypted configuration file."""
        try:
            config_file = Path("config/encrypted_tokens.json")
            if config_file.exists():
                # This would require implementing encryption/decryption
                # For now, return None to indicate not implemented
                pass
        except Exception as e:
            logger.debug("Encrypted config access failed: %s", e)
        return None
    
    def _get_from_env_file(self, token_name: str) -> Optional[str]:
        """Get token from .env file."""
        try:
            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f"{token_name}="):
                            return line.split('=', 1)[1].strip('"\'')
        except Exception as e:
            logger.debug("Env file access failed: %s", e)
        return None
    
    def _get_from_prompt(self, token_name: str) -> Optional[str]:
        """Get token from interactive prompt."""
        try:
            prompt = f"Enter {token_name}: "
            if "SECRET" in token_name or "TOKEN" in token_name or "PASSWORD" in token_name:
                return getpass.getpass(prompt)
            else:
                return input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def _store_in_keychain(self, token_name: str, token: str) -> bool:
        """Store token in system keychain."""
        if not KEYRING_AVAILABLE:
            return False
        
        try:
            keyring.set_password(self.service_name, token_name, token)
            logger.info("Token '%s' stored in keychain", token_name)
            return True
        except Exception as e:
            logger.warning("Failed to store token in keychain: %s", e)
            return False
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        config = {
            'motherduck_token': self.get_token('MOTHERDUCK_TOKEN'),
            'aws_access_key_id': self.get_token('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': self.get_token('AWS_SECRET_ACCESS_KEY'),
            'aws_region': self.get_token('AWS_DEFAULT_REGION') or 'us-east-1',
            'database_path': self.get_token('DUCKDB_DATABASE_PATH') or 'space_analytics.db',
            'warehouse_path': self.get_token('ICEBERG_WAREHOUSE_PATH') or 'data/iceberg_warehouse'
        }
        return {k: v for k, v in config.items() if v is not None}
    
    def setup_interactive(self):
        """Interactive setup for first-time configuration."""
        print("🚀 Space Analytics Configuration Setup")
        print("=" * 50)
        
        # Required tokens
        required_tokens = {
            'MOTHERDUCK_TOKEN': 'MotherDuck authentication token (optional)',
            'AWS_ACCESS_KEY_ID': 'AWS Access Key for S3/Iceberg (optional)',
        }
        
        for token_name, description in required_tokens.items():
            current = self.get_token(token_name)
            if current:
                print(f"✅ {token_name}: Already configured")
            else:
                print(f"\n📝 {description}")
                value = self._get_from_prompt(token_name)
                if value:
                    self._store_in_keychain(token_name, value)
        
        print("\n✅ Configuration setup complete!")
        return self.get_all_config()


# Global instance
secure_config = SecureConfigManager()


def get_secure_token(token_name: str, required: bool = False) -> Optional[str]:
    """Convenience function to get a token securely."""
    return secure_config.get_token(token_name, required=required)


# Example usage functions
def get_motherduck_token() -> Optional[str]:
    """Get MotherDuck token from secure sources."""
    return get_secure_token('MOTHERDUCK_TOKEN')


def get_aws_credentials() -> Dict[str, Optional[str]]:
    """Get AWS credentials from secure sources."""
    return {
        'access_key_id': get_secure_token('AWS_ACCESS_KEY_ID'),
        'secret_access_key': get_secure_token('AWS_SECRET_ACCESS_KEY'),
        'region': get_secure_token('AWS_DEFAULT_REGION') or 'us-east-1'
    }


if __name__ == "__main__":
    # Interactive setup
    config_manager = SecureConfigManager()
    config_manager.setup_interactive()