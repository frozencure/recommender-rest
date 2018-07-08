import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Recommendation(db.Model):
    userId = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    recommendations = db.Column(db.VARCHAR, nullable=False)



    def getRecommendationsAsArray(self, n=10):
        recommendationsArr = json.loads(self.recommendations)
        result = []
        for recommendation in recommendationsArr[:n]:
            questionId = recommendation['Question']
            result.append(questionId)
        return result

    def __repr__(self):
        return 'User:' + str(self.userId) + ', Recommendations:' + self.recommendations
