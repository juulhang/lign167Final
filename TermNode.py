class TermNode:
    '''
    the purpose of this is class is to store words, their type, and a dictionary of every word linked to it
    attributes:
        name: a string of what it is named
        kind: the type of the object (drug, group, etc)
        links: a list containing objects (name, kind, links) that this object is linked to
    '''

    def __init__(self, name: str, kind: str, links: list = list()):
        '''
        initialize the class with name, type, and empty dictionary
        '''

        self.name = name
        self.kind = kind
        self.links = links

    def getName(self) -> str:
        '''
        gets the name of the term
            returns: self.name, the name of the term
        '''

        return self.name

    def setName(self, name) -> None:
        '''
        sets name to new input
        '''

        self.name = name

    def getType(self) -> str:
        '''
        get the type of the term
            returns: self.kind, the type of the term
        '''

        return self.kind

    def setType(self, kind) -> None:
        '''
        sets type to new input
        '''

        self.kind = kind

    def getLinks(self) -> list:
        '''
        get the list of every term this term is linked to
            returns: self.links, a list of ddistrucs which are linked to this term
        '''

        return self.links

    def getLinkNames(self) -> list:
        '''
        get a list of names (in str type) of terms that this term is linked to
            returns: names, a list of .getName() for every term in self.links
        '''

        names = [ node.getName() for node in self.links ]

        return names

    def getLinkDict(self) -> dict:
        '''
        get a dict of {name:obj} of terms that this term is linked to
        
            returns: names, a dict wherein the keys are names of terms and the value is the term 
                        object itself for every term in self.links
        '''

        names = { node.getName() : node for node in self.links }

        return names

    def addLink(self, name, kind) -> None:
        '''
        add a link, such that the two terms are linked
        '''

        self.links.append(TermNode(name, kind))

