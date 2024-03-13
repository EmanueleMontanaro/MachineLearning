class Partecipant:
    total = 5

    def __init__(self,name,surname,nationality,scores):
        self.name = name
        self.surname = surname
        self.nationality = nationality
        self.scores = scores
    
    def scoreCalculator(self):
        score = 0
        for value in self.scores:
            score += value
        return score/self.total
    
    def printFullName(self):
        print(self.name + ' ' + self.surname)
    
#import sys
#fileName = sys.argv[1]
fileName = 'scores.txt'
fin = open('fileName','r')

partecipantsList = []

for line in fin:
    string = fin.readline()
    array = string.split(' ')
    person = Partecipant(array[0],array[1],array[2],array[3:8])
    partecipantsList.append(person)

for person in partecipantsList:
    person.printFullName()
