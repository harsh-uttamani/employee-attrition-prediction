import unittest
from app import app, db, companies_collection, users_collection
from werkzeug.security import generate_password_hash

class AppTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        # Clean up database for tests
        companies_collection.delete_many({})
        users_collection.delete_many({})

    def test_signup_and_login(self):
        # 1. Test unauthenticated access redirects to login
        res = self.client.get('/')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers.get('Location'))

        # 2. Test signup
        signup_data = {
            'company_name': 'Test Corp',
            'company_email': 'contact@testcorp.com',
            'admin_name': 'Admin User',
            'username': 'admin@testcorp.com',
            'password': 'password123',
            'confirm_password': 'password123'
        }
        res = self.client.post('/signup', data=signup_data)
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers.get('Location'))

        # Verify DB
        self.assertEqual(companies_collection.count_documents({}), 1)
        self.assertEqual(users_collection.count_documents({}), 1)

        # 3. Test login
        login_data = {
            'username': 'admin@testcorp.com',
            'password': 'password123'
        }
        res = self.client.post('/login', data=login_data)
        self.assertEqual(res.status_code, 302)
        self.assertIn('/', res.headers.get('Location'))

        # 4. Test authenticated access
        with self.client.session_transaction() as sess:
            # Manually simulating what happens after login for the client if needed
            pass
            
        # The test_client keeps cookies so the session should be valid now
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'View Methodology', res.data)
        self.assertIn(b'Logout', res.data)
        
        # 5. Test methodology page accessible
        res = self.client.get('/methodology')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Employee Attrition Prediction System', res.data)
        
        # 6. Test logout
        res = self.client.get('/logout')
        self.assertEqual(res.status_code, 302)
        self.assertIn('/login', res.headers.get('Location'))
        
        # Should be protected again
        res = self.client.get('/')
        self.assertEqual(res.status_code, 302)

if __name__ == '__main__':
    unittest.main()
