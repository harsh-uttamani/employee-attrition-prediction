import requests
from bs4 import BeautifulSoup
import string
import random

BASE_URL = "http://127.0.0.1:5000"
session = requests.Session()

def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def test_flow():
    company_name = f"TestCorp_{generate_random_string()}"
    company_email = f"admin@{company_name.lower()}.com"
    username = company_email
    password = "SecurePassword123!"
    
    print(f"[*] Testing Signup for {company_name}")
    signup_data = {
        "company_name": company_name,
        "company_email": company_email,
        "admin_name": "Admin User",
        "username": username,
        "password": password,
        "confirm_password": password
    }
    
    res = session.post(f"{BASE_URL}/signup", data=signup_data)
    print(f"[+] Signup status: {res.status_code}")
    
    # After signup, it should redirect to login (status 200 after redirect)
    if "Login" not in res.text:
        print("[-] Did not land on login page after signup.")
    
    print(f"[*] Testing Login for {username}")
    login_data = {
        "username": username,
        "password": password
    }
    res = session.post(f"{BASE_URL}/login", data=login_data)
    print(f"[+] Login status: {res.status_code}")
    
    # After login, should redirect to home page, which contains 'Assess Retention Risk'
    if "Assess Retention Risk" in res.text or "Welcome" in res.text:
        print("[+] Successfully logged in and reached home page!")
    else:
        print("[-] Failed to reach home page after login.")
        if "Invalid credentials" in res.text:
            print("[-] Error: Invalid credentials.")
    
    print("[*] Testing Logout")
    res = session.get(f"{BASE_URL}/logout")
    print(f"[+] Logout status: {res.status_code}")
    
    if "Login" in res.text:
        print("[+] Successfully logged out and reached login page.")
    else:
        print("[-] Failed to reach login page after logout.")

    # Try accessing protected page
    print("[*] Testing protected page access after logout")
    res = session.get(f"{BASE_URL}/")
    if "Login" in res.text:
        print("[+] Protected page redirected to Login as expected.")
    else:
        print("[-] Protected page was accessible without login!")

if __name__ == "__main__":
    test_flow()
