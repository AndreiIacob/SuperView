#!flask/bin/python
from flask import Flask, jsonify
import topicExtraction

app = Flask(__name__)

@app.route('/api/topics/all/<str:business_id>', methods=['GET'])
def get_topics(business_id):
    topics = topicExtraction.get_topics(business_id)
    return jsonify( topics)

if __name__ == '__main__':
    app.run(debug=True)