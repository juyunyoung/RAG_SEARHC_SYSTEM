import bcrypt

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # bcrypt.checkpw() 함수는 입력된 비밀번호와 해시된 비밀번호를 비교하여 일치하는지 확인합니다
    # 첫번째 인자: 검증할 원본 비밀번호를 바이트로 인코딩
    # 두번째 인자: 저장된 해시 비밀번호를 바이트로 인코딩
    # 반환값: 비밀번호가 일치하면 True, 불일치하면 False
    
    return bcrypt.checkpw(
        password.encode('utf-8'),
        hashed_password.encode('utf-8')
    ) 