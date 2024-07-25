from trial_something import create_app
from OpenSSL import SSL 

app = create_app()

if __name__ == '__main__':
    def divide_by_100(x):
        return x/100
    
    # Define SSL certificate and key file paths 
    CERT_FILE = "/path/to/cert.pem" 
    KEY_FILE = "/path/to/key.pem" 

    # Create SSL context 
    context = SSL.Context(SSL.PROTOCOL_TLSv1_2) 
    context.load_cert_chain(CERT_FILE, KEY_FILE) 

    app.run(debug=True, host='165.227.32.65', port=80, ssl_context=context) 



    