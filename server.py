import http.server
import json
from ocr import OCRNeuralNetwork
import numpy as np

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
HIDDEN_NODE_COUNT = 15

# Load data samples and labels into matrix
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter = ',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

# Convert from numpy ndarrays to python lists
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

# If a neural network file does not exist, train it using all 5000 existing data samples.
# Based on data collected from neural_network_design.py, 15 is the optimal number
# for hidden nodes
nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, list(range(5000)));

class JSONHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        response_code = 200
        response = {}
        varLen = int(self.headers.get('Content-Length'))
        content = self.rfile.read(varLen);
        payload = json.loads(content);

        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        elif payload.get('predict'):
            try:
                response = {"type":"test", "result":nn.predict(str(payload['image']))}
            except Exception as e:
                print(f"Error during prediction: {e}")
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))
        return

if __name__ == '__main__':
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        print(f"Starting server on http://{HOST_NAME}:{PORT_NUMBER}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Server error")
    finally:
        httpd.server_close()