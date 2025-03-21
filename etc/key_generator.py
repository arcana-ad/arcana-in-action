import secrets
import string


def generate_key(length=64):
    alphabet = string.ascii_letters + string.digits
    key = "".join(secrets.choice(alphabet) for _ in range(length))
    return key


key = generate_key()
print(key)
