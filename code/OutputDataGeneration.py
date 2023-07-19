from pySurro import *
import numpy
import pandas
from ast import literal_eval
import math
from tqdm import tqdm

# pandas.set_option("display.max_colwidth", 500)

class FibreRowConstructionData:
    def __init__(self, sampleRectangleWidth, sampleRectangleHeight, inputdataRow, filamentNumber, spatialDescretization):
        self.inputdataRow = inputdataRow
        self.sampleRectangleWidth = sampleRectangleWidth * 1e-3
        self.sampleRectangleHeight = sampleRectangleHeight * 1e-3
        self.filamentNumber = filamentNumber
        self.spatialDescretization = spatialDescretization
        #stores sample rectangle's coordinates lower left, width and height
        self.sampleRectangle = numpy.zeros(4)
        self.H = 0.0
        self.W = 0.0
        self.h = 0.0
        self.w = 0.0
        self.SetSimulationWindowParameters()
        
    #calculates the Area of the window to be simulated in the SURRO tool
    def SetSimulationWindowParameters(self):
        # calculate according to fiber mass distribution
        self.h = math.ceil(max(3*self.inputdataRow['Sigma_2'] + 5, 10)) * 1e-3
        self.w = math.ceil(max(5*self.inputdataRow['Sigma_1'] + 15, 100)) * 1e-3
        self.H = 2*self.h + self.sampleRectangleHeight;
        self.W = 2*self.w + self.sampleRectangleWidth;
        self.sampleRectangle[0] = self.w 
        self.sampleRectangle[1] = self.h
        self.sampleRectangle[2] = self.sampleRectangleWidth
        self.sampleRectangle[3] = self.sampleRectangleHeight
        
    #returns sample rectangle's coordinates lower left, width and height
    def GetSampleRectangle(self):
        return self.sampleRectangle
      
    # creates and returns construction data for the current row of input parameters
    def GetConstructionData(self):
        sgt  = SpinGroupRow
        fibreLength = self.W / self.inputdataRow['BeltSpinRatio']
        # number of points in fibre
        fibreNumber = math.ceil(fibreLength / self.spatialDescretization)
        disc = DiscretizationData(nTrackCurves = self.filamentNumber, N = fibreNumber, ds = self.spatialDescretization)
        # number of spin positions
        distance = 1 / self.inputdataRow['SpinPositionsPerMeterInverse']
        spinPositionNumber = math.ceil((self.H / distance) + 1)
#         spinPositionNumber = int((self.H / (0.1 * 1e-3)) + 1)
        pos = SpinPositions(md2 = 0, cd2 = self.H, nSpinPositions12 = spinPositionNumber)
        group = GroupData("FibreGroup", disc, pos)
        trackCurve = TrackCurveData(self.inputdataRow['BeltSpinRatio'], 1) 
        material = MaterialData(1.1e-07, 1e-5)
        filament = FilamentData(sigma1=self.inputdataRow['Sigma_1'] * 1e-3, sigma2=self.inputdataRow['Sigma_2']* 1e-3, 
                                phi=0, A=self.inputdataRow['A']) 
        randomSeeds = self.inputdataRow['RandomSeeds']
        cd = ConstructionData(sgt, group, trackCurve, material, filament, randomSeeds) 
        return cd
    
    # Compute CV value
    def ComputeCV(self, m):
        return 100*numpy.std(m)/numpy.mean(m)
        
    
def ConvetToArray(row):
    return numpy.array(literal_eval(row['RandomSeeds']))



# processes the database row by row to create CV values at different resolutions as output
# and saves the output along with the input as csv
def GenerateOutput(row):
    fibreRowData = FibreRowConstructionData(500, 250, row, 1, 2.5e-5)
    cd = fibreRowData.GetConstructionData()
    sampleRect = fibreRowData.GetSampleRectangle()
    cv_list = list()
    raster = computeBaseWeight([cd], sampleRect[0], sampleRect[1], sampleRect[2], sampleRect[3], 
                               [0.5*1e-3, 1*1e-3, 2*1e-3, 5*1e-3, 10*1e-3, 20*1e-3, 50*1e-3], batchSize = 15)
    for i in raster:
        cv = fibreRowData.ComputeCV(i.data)
        cv_list.append(cv)
    return cv_list

def main():
    inputDatabase = pandas.read_csv("/p/tv/FIDYST2/VinyM/Surro/src/pySurro/examples/inputbatch4/SURRO_input_database_reduced_2.csv")
    inputDatabase['RandomSeeds'] = inputDatabase.apply(ConvetToArray, axis = 1)
    tqdm.pandas()
#     inputDatabase['CV_Value'] = inputDatabase.progress_apply(GenerateOutput, args=(resolutions,) , axis = 1)
    inputDatabase['CV_Value'] = inputDatabase.progress_apply(GenerateOutput, axis = 1)
    inputDatabase.to_csv('output4thbatch/SURRO_output_database_complete_2.csv', index=False)

if __name__=='__main__':
    main()
