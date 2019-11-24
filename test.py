import pandas as pd
import numpy as np
import util
import draw
import matplotlib.pyplot as plt

class Father(object):
    def __init__(self, name):
        self.name=name
        print ( "name: %s" %( self.name) )
    def getName(self, extra):
        return 'Father ' + self.name + extra

class Son(Father):
    def getName(self, extra = 's'):
        return 'Son '+self.name + extra

if __name__=='__main__':
    son=Son('runoob')
    print ( son.getName() )