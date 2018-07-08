from scipy import sparse
import pandas as pd
from sklearn import metrics
import numpy as np

class SparseDataframe:

    """
    Class that handles the conversion and analysis of a data-set which contains user votes
    that are related to specific items. The data-set will be mapped as a pandas dataframe,
    which will be then converted in a sparse matrix, for more efficient memory management.
    attributes:
        dataframe: pandas dataframe from wich sparse matrix will be built.
        columns: list of the headers of the dataframe -> 1: Items, 2: Users, 3: Votes
                A row of the df contains: A vote an user casted on a item.
        uniqueUsers: list of all unique users, should be first column
        uniqueItems: list of all unique items
        itemVoteCounts: list of each item and how many votes it has
        userVoteCounts: list of each user and how many votes he casted
        hasItemAsRows: boolean variable which is used to specify the matrix setup: (items, users) or (users, items)
        csrMatrix: a sparse csr matrix that is built from the dataframe(check scipy.sparse)
    """
    def __init__(self, dataframe=None, greaterThan=0, csvPath=None, hasItemsAsRows=True, filePath=None):
        """Will create a filtered dataframe by removing low voted items"""
        if csvPath is not None or dataframe is not None:
            if csvPath is not None:
                self.dataframe = pd.read_csv(csvPath, compression='gzip')
            if dataframe is not None:
                self.dataframe = dataframe
            self.dataframe['Votes'] = np.ones(shape=(len(self.dataframe.index)))
            self.columns = list(self.dataframe)
            self.itemVoteCounts = dataframe[dataframe.columns[0]].value_counts()
            self.userVoteCounts = dataframe[dataframe.columns[1]].value_counts()
            self.removeLowUsers(greaterThan)
            self.removeLowVotes(greaterThan)
            """Will populate the csr matrix and itemVoteCounts attributes"""
            self.uniqueUsers = self.__setUniqueUsers()
            self.uniqueItems = self.__setUniqueItems()
            self.hasItemAsRows = hasItemsAsRows
            self.csrMatrix = self.__setSparseMatrix()
        elif filePath is not None:
            self.load_sparse_csr(filepath=filePath)
            self.hasItemAsRows = hasItemsAsRows


    def __setUniqueUsers(self):
        un_users = self.dataframe[self.dataframe.columns[1]].unique().tolist()
        # un_users.sort()
        return un_users

    def __setUniqueItems(self):
        un_items = self.dataframe[self.dataframe.columns[0]].unique().tolist()
        # un_items.sort()
        return un_items

    def __getDataAsList(self):
        data = self.dataframe[self.dataframe.columns[2]].tolist()
        return data

    def __setSparseMatrix(self):
        if self.hasItemAsRows:
            rows = self.dataframe[self.dataframe.columns[0]].astype(pd.api.types.CategoricalDtype(categories=self.uniqueItems)).cat.codes
            columns = self.dataframe[self.dataframe.columns[1]].astype(pd.api.types.CategoricalDtype(categories=self.uniqueUsers)).cat.codes
        else:
            columns = self.dataframe[self.dataframe.columns[0]].astype(
                pd.api.types.CategoricalDtype(categories=self.uniqueItems)).cat.codes
            rows = self.dataframe[self.dataframe.columns[1]].astype(
                pd.api.types.CategoricalDtype(categories=self.uniqueUsers)).cat.codes
        data = self.__getDataAsList()
        if self.hasItemAsRows:
            csrMatrix = sparse.csr_matrix((data, (rows, columns)), shape=(len(self.uniqueItems), len(self.uniqueUsers)))
        else:
            csrMatrix = sparse.csr_matrix((data, (rows, columns)), shape=(len(self.uniqueUsers), len(self.uniqueItems)))
        return csrMatrix

    def getItemVoteCount(self, itemId):
        return self.itemVoteCounts[itemId]

    def getUserIdFromIndex(self, userIndex):
        return self.uniqueUsers[userIndex]

    def getItemIdFromIndex(self, itemIndex):
        return self.uniqueItems[itemIndex]

    def getUserIndexById(self, userId):
        return self.uniqueUsers.index(userId)

    def getItemIndexById(self, itemId):
        if itemId in self.uniqueItems:
            return self.uniqueItems.index(itemId)
        else:
            return False

    def getItemsIndexByUser(self, userId):
        userIndex = self.getUserIndexById(userId)
        if self.hasItemAsRows:
            return self.csrMatrix.getcol(userIndex).nonzero()[0]
        else:
            return self.csrMatrix.getrow(userIndex).nonzero()[1]

    def __getUsersIndexByItem(self, itemId):
        itemIndex = self.getItemIndexById(itemId)
        if self.hasItemAsRows:
            return self.csrMatrix.getrow(itemIndex).nonzero()[1]
        else:
            return self.csrMatrix.getcol(itemIndex).nonzero()[0]

    def getItemIdsByUser(self, userId):
        itemsIndexes = self.getItemsIndexByUser(userId)
        itemsIds = []
        for index in itemsIndexes:
            itemsIds.append(self.getItemIdFromIndex(index))
        return itemsIds

    def getUserIdsByItem(self, itemId):
        userIndexes = self.__getUsersIndexByItem(itemId)
        userIds = []
        for index in userIndexes:
            userIds.append(self.getUserIdFromIndex(index))
        return userIds

    def removeLowVotes(self, smallerThan):
        filteredCounts = self.itemVoteCounts[self.itemVoteCounts > smallerThan]
        self.dataframe = self.dataframe[self.dataframe[self.dataframe.columns[0]].isin(filteredCounts.index.tolist())]

    def removeLowUsers(self, smallerThan):
        filteredCounts = self.userVoteCounts[self.userVoteCounts > smallerThan]
        self.dataframe = self.dataframe[self.dataframe[self.dataframe.columns[1]].isin(filteredCounts.index.tolist())]

    def getTopItemsCosineSim(self, postId, top=5):
        if self.getItemIndexById(postId) == False:
            return False
        index = self.getItemIndexById(postId)
        if self.hasItemAsRows:
            indexVector = self.csrMatrix[index, :]
            cosSim = metrics.pairwise.cosine_similarity(indexVector, self.csrMatrix, dense_output=False)
        else:
            indexVector = self.csrMatrix[:, index]
            cosSim = metrics.pairwise.cosine_similarity(indexVector.T, self.csrMatrix.T, dense_output=False)
        similaritiesContainer = cosSim.toarray()
        similarities = similaritiesContainer[0]
        ind = np.argpartition(similarities, -top - 1)[-top - 1:]
        indexSim = [(i, similarities[i]) for i in ind]
        indexSim = [value for value in indexSim if value[0] != self.getItemIndexById(postId)]
        indexSim.sort(key=lambda x: x[1], reverse=True)
        idSim = {}
        for elem in indexSim:
            id = self.getItemIdFromIndex(elem[0])
            acc = elem[1]
            idSim[id] = acc
        return idSim

    def save_sparse_csr(self, filepath):
        np.savez(filepath, items=self.uniqueItems, users=self.uniqueUsers, data=self.csrMatrix.data, indices=self.csrMatrix.indices,
                 indptr=self.csrMatrix.indptr, shape=self.csrMatrix.shape)

    def load_sparse_csr(self, filepath):
        loader = np.load(filepath)
        self.uniqueUsers = loader['users'].tolist()
        self.uniqueItems = loader['items'].tolist()
        self.csrMatrix = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])

    def load_sparse_csr(self, filepath):
        # if filepath is str:
        #     loader = np.load(filepath)
        # else:
        #     loader = filepath
        loader = np.load(filepath)
        self.uniqueUsers = loader['users'].tolist()
        self.uniqueItems = loader['items'].tolist()
        self.csrMatrix = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                          shape=loader['shape'])


# def getResourcesPath():
#     repo_path = str(pathlib.Path(os.getcwd()).parent)
#     return repo_path + '/resources/'
#
# sparseDf = SparseDataframe(filePath=getResourcesPath() + 'matrix.npz')
# print(type(sparseDf.uniqueItems))
# print(type(sparseDf.uniqueUsers))
# print(type(sparseDf.csrMatrix))
# time.sleep(30)
# print(sparseDf.getTopItemsCosineSim(postId=24002369))