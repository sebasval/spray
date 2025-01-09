from typing import Dict, Optional, List
from fastapi import HTTPException
from .models import User, UserCreate
from .security import get_password_hash

# Base de datos en memoria (diccionario)
users_db: Dict[str, User] = {}

class DatabaseManager:
    """
    Manejador de la base de datos de usuarios
    """
    @staticmethod
    def get_user(email: str) -> Optional[User]:
        """
        Obtiene un usuario por su email
        """
        return users_db.get(email)
    
    @staticmethod
    def create_user(user: UserCreate) -> User:
        """
        Crea un nuevo usuario
        Lanza una excepción si el email ya existe
        """
        if user.email in users_db:
            raise HTTPException(
                status_code=400,
                detail="El email ya está registrado"
            )
        
        # Crear el usuario con la contraseña hasheada
        db_user = User(
            email=user.email,
            username=user.username,
            hashed_password=get_password_hash(user.password)
        )
        users_db[user.email] = db_user
        return db_user
    
    @staticmethod
    def list_users() -> List[User]:
        """
        Retorna la lista de todos los usuarios
        """
        return list(users_db.values())
    
    @staticmethod
    def delete_user(email: str) -> bool:
        """
        Elimina un usuario por su email
        Retorna True si el usuario fue eliminado, False si no existía
        """
        if email in users_db:
            del users_db[email]
            return True
        return False
    
    @staticmethod
    def update_last_login(email: str) -> None:
        """
        Actualiza la fecha del último login de un usuario
        """
        from datetime import datetime
        if email in users_db:
            users_db[email].last_login = datetime.now()

    @staticmethod
    def count_users() -> int:
        """
        Retorna el número total de usuarios
        """
        return len(users_db)
    
    @staticmethod
    def is_email_registered(email: str) -> bool:
        """
        Verifica si un email ya está registrado
        """
        return email in users_db