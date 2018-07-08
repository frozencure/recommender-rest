# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging
from flask import Flask, jsonify, request
from processing.SparseDataframe import SparseDataframe
from processing.ModelContainer import ModelContainer
from Models import db, Recommendation
import os


app = Flask(__name__)
try:
    databaseCredetials = os.environ['SQLALCHEMY_DATABASE_URI']
except Exception:
    databaseCredetials = None
if databaseCredetials is not None:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://admin:admin@localhost/stackoverflowDb'
db.init_app(app)
sparseDfSmall = SparseDataframe(filePath='resources/sparseDf.npz', hasItemsAsRows=False)
sparseDfBig = SparseDataframe(filePath='resources/sparseDfBig.npz', hasItemsAsRows=False)
modelContainer = ModelContainer(sparseDf=sparseDfSmall)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    # redis.flushall()
    return 'Stackoverflow Recommender API'

@app.route('/api/v1.0/similarQuestions', methods=['GET'])
def getSimilarQuestions():
    questionId = int(request.args.get('questionId', None))
    top = int(request.args.get('top', 5))
    similarQuestions = sparseDfBig.getTopItemsCosineSim(questionId, top)
    return jsonify({'similarQuestions':similarQuestions})

@app.route('/api/v1.0/topRecommendationsBPR', methods=['GET'])
def getTopRecommendationsBPR():
    userId = int(request.args.get('userId', None))
    n = int(request.args.get('top', 10))
    topRecommendations = modelContainer.topRecommendationsBPR(userId=userId, n=n)
    return jsonify({'topRecommendations':topRecommendations})


@app.route('/api/v1.0/topRecommendationsALS', methods=['GET'])
def getTopRecommendationsASL():
    userId = int(request.args.get('userId', None))
    n = int(request.args.get('top', 10))
    topRecommendations = modelContainer.topRecommendationsALS(userId=userId, n=n)
    return jsonify({'topRecommendations': topRecommendations})


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
