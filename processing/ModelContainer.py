import pickle
import numpy as np
from Models import Recommendation

class ModelContainer:

    def __init__(self, sparseDf):
        self.model = pickle.load(open('resources/model.pickle', 'rb'))
        self.sparseDf = sparseDf

    def topRecommendationsBPR(self, userId, n=10):
        excludeArr = np.array(self.sparseDf.getItemsIndexByUser(userId=userId))
        itemIds = np.arange(len(self.sparseDf.uniqueItems))
        # itemIds = np.setdiff1d(itemIds, excludeArr)
        predictions = self.model.predict(user_ids=self.sparseDf.getUserIndexById(userId), item_ids=itemIds)
        predictionIndeces = np.argsort(predictions)[::-1][:n]
        resultArr = []
        for index in predictionIndeces:
            resultArr.append(self.sparseDf.getItemIdFromIndex(index))
        return resultArr


    def topRecommendationsALS(self, userId, n=10):
        recommendations = Recommendation.query.filter_by(userId=userId).first()
        return recommendations.getRecommendationsAsArray(n=n)