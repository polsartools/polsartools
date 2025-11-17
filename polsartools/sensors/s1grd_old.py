from osgeo import gdal
gdal.UseExceptions()
import os
import numpy as np
import glob
from xml.etree import cElementTree as ElementTree
from scipy.interpolate import LinearNDInterpolator as interpnd
import warnings
warnings.filterwarnings('ignore')

from .basic_func import read_bin, clipRaster, write_tif

class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})
                
def sigmaCal(incalXml,rows,cols):
    """ SIGMA0 CALIBRATION LUT"""
    tree = ElementTree.parse(incalXml)
    root = tree.getroot()
    # xmldict = XmlDictConfig(root)

    caliRoot = root.find('calibrationVectorList')
    # nVectors  = int(caliRoot.items()[0][1])
    xx = [];
    yy = [];
    zz = [];
#     for child in caliRoot.getchildren():
    for child in list(caliRoot):
        line  = int(child.find('line').text)
        pixel = list(map(int, child.find('pixel').text.split()))
        nPixel = int(child.find('pixel').items()[0][1])
        sigmaNought = list(map(float, child.find('sigmaNought').text.split()))
        xx = xx + pixel
        yy = yy + [line]*nPixel
        zz = zz + sigmaNought
        
    npt = len(zz)
    coord = np.hstack((np.array(xx).reshape(npt,1),np.array(yy).reshape(npt,1)))
    sigma  = np.array(zz).reshape(npt,1)
    interpfn1  = interpnd(coord,sigma)
    fullX, fullY = np.meshgrid(list(range(cols)), list(range(rows)))
    return interpfn1(fullX, fullY)


def s1grd_sigma0(dataFolder):

	# dataFolder = r"./s1"
	# clip Extent North, South, East, West
	# maxy, miny, maxx, minx = 49.4,  49.3, -98.9, -99.1
	outFolder = dataFolder+'/processed'
	if not os.path.exists(outFolder) : os.mkdir(outFolder)

	for pol in ['vv']:
		inRaster = glob.glob(dataFolder+r"/measurement/*"+pol+"*.tiff")[0]
		incalXml = glob.glob(dataFolder+r"/annotation/calibration/calibration-*grd-"+pol+"-*.xml")[0]
		gcpXml = glob.glob(dataFolder+r"/annotation/*grd-"+pol+"-*.xml")[0]

		# Radiomertic calibration
		ds = gdal.Open(inRaster)
		cols,rows =  ds.RasterXSize,ds.RasterYSize

		sigmaIntrp = sigmaCal(incalXml,rows,cols)
		#             calSigma0 = 10*np.log10(ds.GetRasterBand(1).ReadAsArray()**2/sigmaIntrp[:,:,0]**2)
		calSigma0 = ds.GetRasterBand(1).ReadAsArray()**2/sigmaIntrp[:,:,0]**2
		calSigma0[calSigma0==-1*np.inf] = np.nan
		tempFile = outFolder+'/temp_cal.tiff'
		write_tif(tempFile,calSigma0,inRaster)


		tree = ElementTree.parse(gcpXml)
		root = tree.getroot()
		xmldict = XmlDictConfig(root)
		gcpGrid = xmldict.get("geolocationGrid")
		values = list(gcpGrid.values())[0].get('geolocationGridPoint')
		gcpList =[] 
		for ii in range(len(values)):
		    gcpList.append([int(values[ii].get('pixel')),int(values[ii].get('line')),
		                  float(values[ii].get('longitude')),float(values[ii].get('latitude')),
		                  float(values[ii].get('height'))])


		# Gecoding 
		tempFile_geo = outFolder+'/temp_cal_geo.tiff'
		dsIn = gdal.Open(tempFile)
		gdal.Translate(tempFile_geo, dsIn, outputSRS='EPSG:4326',
		    GCPs=[gdal.GCP(x, y, z, pixel, line) for (pixel, line, x, y, z) in gcpList],)
		dsIn=None