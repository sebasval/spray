from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class User(BaseModel):
    """Modelo de usuario para la base de datos"""
    email: EmailStr
    username: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = datetime.now()
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    """Modelo para crear un nuevo usuario"""
    email: EmailStr
    username: str
    password: str

class Token(BaseModel):
    """Modelo para el token de autenticación"""
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """Modelo para los datos contenidos en el token"""
    email: Optional[str] = None