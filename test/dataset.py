from collections import Counter

import numpy as np
import torch
import torch.utils
import torch.utils.data
import itertools
from tqdm import tqdm
import hdf5storage
import scipy
from utils.dataset import open_file
import h5py

DATASETS_CONFIG = {
    'PaviaC': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
        #         'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat',
    },
    'Salinas': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
        #          'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'],
        'img': 'Salinas_corrected.mat',
        'gt': 'Salinas_gt.mat',
    },
    'PaviaU': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
        #          'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat',
    },
    'ksc': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
        #          'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
        'img': 'ksc.mat',
        'gt': 'ksc_gt.mat',
    },
    'IndianPines': {
        # 'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        #         'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
        'img': 'Indian_pines_corrected.mat',
        'gt': 'Indian_pines_gt.mat'
    },
    'Botswana': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat',
    },
    'xiongan_one': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_one.mat',
        'gt': 'xiongan_one_gt.mat',
    },
    'xiongan_two': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_two.mat',
        'gt': 'xiongan_two_gt.mat',
    },
    'xiongan_four': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_four.mat',
        'gt': 'xiongan_four_gt.mat',
    },
    'xiongan_three': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'xiongan_three.mat',
        'gt': 'xiongan_three_gt.mat',
    },
    'houston2013': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'houston.mat',
        'gt': 'houston.mat',
    },
    'houston2018': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.hdr",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
    },
    'WHU-Hi-HanChuan': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_HanChuan.mat",
        "gt": "WHU_Hi_HanChuan_gt.mat",
    },
    'whu_honghu': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_HongHu.mat",
        "gt": "WHU_Hi_HongHu_gt.mat",
    },
    'whu_longkou': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_LongKou.mat",
        "gt": "WHU_Hi_LongKou_gt.mat",
    },
    'MUUFLGulfport': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        "img": "WHU_Hi_LongKou.mat",
        "gt": "WHU_Hi_LongKou_gt.mat",
    },
    'sd': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'houston.mat',
        'gt': 'houston.mat',
   },
    'huanghe': {
        # 'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
        #         'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'huanghekou_data.mat',
        'gt': 'huanghekou_samples.mat',
   }

}

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def get_dataset(dataset_name, need_pca=False, target_folder="/"):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    
    palette = None
    
    print('when run the def get_dataset, the dataset_name is :',dataset_name)

    folder = "/home/t1/DL_Code/DATASET/" + dataset_name + "/"

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')
        img = img['pavia']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']
        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]
        ignored_labels = [0]

    elif dataset_name == 'sd':
        # Load the image
        img = open_file('./Datasets/' + 'add_fog_dataset.mat')
        img = img['Iw']
        rgb_bands = (55, 41, 12)
        gt = open_file('./Datasets/PaviaU/' + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['hunkown','Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]
    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')
        img = img['paviaU']
        rgb_bands = (55, 41, 12)
        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['hunkown','Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]
    elif dataset_name == 'Salinas':
        img = open_file(folder + 'Salinas_corrected.mat')['salinas_corrected']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth', 'Stubble', 'Celery',
                        'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk',
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk',
                        'Lettuce_romaine_7wk', 'Vinyard_untrained', 'Vinyard_vertical_trellis']
        ignored_labels = [0]
    elif dataset_name == 'IndianPines':
        # Load the image  1992年Indian Pines数据集
        data = scipy.io.loadmat(folder + 'Indian_pines.mat')

        for key in data.keys():
            print(key)
        img = open_file(folder + 'Indian_pines_corrected.mat')['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'Indian_pines.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0] 
    elif dataset_name == 'Chikusei':
        with h5py.File(folder + 'HyperspecVNIR_Chikusei_20140729.mat', 'r') as f:
            chikusei_data = np.array(f['chikusei']) 
            img = chikusei_data.transpose(2, 1, 0) 
        label_data = hdf5storage.loadmat(folder + 'HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat')
        gt = label_data['GT'][0][0][0]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        label_values = ["Undefined", "Farmland", "Bare soil", "Water", "Road", "Ground shadow", "Individual tree",
            "Grass", "Conifers", "Broadleaf", "Building", "Asphalt", "Rich grass",  
            "Poor grass",  "Artificial turf", "Greenhouse",  "Bare rock", "Tennis court",  "Parking lot",    # 18
]

        ignored_labels = [0] 
    elif dataset_name == 'huanghe':
        # Yellow River Estuary scene
        print("1")
        image = hdf5storage.loadmat((folder + "huanghekou_data.mat"))

        img = image['huanghekou_data']

        rgb_bands = (30, 15, 8)  # AVIRIS sensor


        label =  hdf5storage.loadmat((folder + "huanghekou_samples.mat"))

        gt = label['huanghekou_gt']


        label_values = ["Undefined", "Salt marsh", "Acquaculture",
                        "Mud flat", "Rice", "Aquatic vegetation",
                        "Seep sea","Freshwater herbaceous marsh"," Shallow sea",
                        "Reed", "Pond", "Buildings", "Suaeda salsa", "Flood plain", "River",
                        "Soybean", "Broomcorn", "Maize", "Locust", "Spartina", "Tamarix", "Intertidal saltwater"]
        ignored_labels = [0]
    
    elif dataset_name == 'MUUFLGulfport':
        # Load the image  1992年Indian Pines数据集
        img = open_file(folder + 'muufl_gulfport_campus_1_hsi.mat')
        img = img['data']
        # img = open_file(folder + 'Indian_pines.mat')
        # img = img['indian_pines']
        rgb_bands = (30, 15, 8)  # AVIRIS sensor
        gt = open_file(folder + 'muufl_gulfport_campus_1_hsi_gt.mat')['gt']
        gt[gt == -1] = 0
        label_values = ["Undefined", "trees", "mostly-grass",
                        "mixed ground surface", "dirt-sand", "road",
                        "water","shadow of buildings"," buildings",
                        "sidewalk", "yellow curb", "cloth panels"]
        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']
        rgb_bands = (75, 33, 15)
        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]
        ignored_labels = [0]
    elif dataset_name == 'ksc':
        # Load the image
        img = open_file(folder + 'ksc.mat')
        img = img['KSC']
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        gt = open_file(folder + 'ksc_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        ignored_labels = [0]
    elif dataset_name == 'houston2013':
        # Load the image
        img = open_file(folder + 'Houston.mat')
        # img = open_file(folder + '2013_IEEE_GRSS_DF_Contest_CASI.tif')
        img = img['Houston']
        rgb_bands = (59, 40, 23)  # AVIRIS sensor
        gt = open_file(folder + 'Houston_gt.mat')['houston_gt']


        label_values = ["Undefined",
                        "Healthy grass",
                        "Stressed grass",
                        "Synthetic grass",
                        "Trees",
                        "Soil",
                        "Water",
                        "Residential",
                        "Commercial",
                        "Road",
                        "Highway",
                        "Railway",
                        "Parking Lot 1",
                        "Parking Lot 2",
                        "Tennis Court",
                        "Running Track",
                        ]
        ignored_labels = [0]
    elif dataset_name == 'houston2018':
        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')[:, :, :-2]
        print(img.shape)
        # quit()
        # gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')#[:,0:4172]
        gt = open_file(folder + 'gt.mat')['houston2018_gt']
        print(gt.shape)
        # quit()
        gt = gt.astype('uint8')
        palette = None
        rgb_bands = (47, 31, 15)

        label_values = ["Unclassified",
                        "Healthy grass",
                        "Stressed grass",
                        "Artificial turf",
                        "Evergreen trees",
                        "Deciduous trees",
                        "Bare earth",
                        "Water",
                        "Residential buildings",
                        "Non-residential buildings",
                        "Roads",
                        "Sidewalks",
                        "Crosswalks",
                        "Major thoroughfares",
                        "Highways",
                        "Railways",
                        "Paved parking lots",
                        "Unpaved parking lots",
                        "Cars",
                        "Trains",
                        "Stadium seats"]
        ignored_labels = [0]
    elif dataset_name == 'sd':
        # Load the image
        # img = open_file(folder + 'houston.mat')
        img = open_file(folder + 'Simu_data.mat')
        img = img['Simu_data']
        rgb_bands = (59, 40, 23)  # AVIRIS sensor
        gt = open_file(folder + 'Simu_label.mat')['Simu_label']
        # img = scipy.io.loadmat(folder + 'Simu_data.mat'))['Simu_data']
        # Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_label.mat'))['Simu_label']

        label_values = ["Undefined",
                        "Healthy grass",
                        "Stressed grass",
                        "Synthetic grass",
                        "Trees",
                        "Soil",
                        ]
        ignored_labels = [0]

        # Data = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_data.mat'))['Simu_data']
        # Label = scipy.io.loadmat(os.path.join(DATA_PATH, 'Simu_label.mat'))['Simu_label']

    elif dataset_name == 'xiongan_one':
        img = open_file(folder + 'xiongan_one.mat')
        img = img['xiongan_one']
        gt = open_file(folder + 'Xiongan_one_gt_changed.mat')
        gt = gt['xiongan_one_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "locust tree",
                        "water",
                        "Corn",
                        "pear",
                        "Soybean",
                        "aspen",
                        "vegetable field",
                        "glassland",
                        "peach tree"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_two':

        img = open_file(folder + 'xiongan_two.mat')
        # print(img)
        img = img['xiongan_two']

        gt = open_file(folder + 'xiongan_two_gt_changed.mat')['xiongan_two_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "paddy",
                        "water",
                        "bare land",
                        "paddy stubble"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_three':

        img = open_file(folder + 'xiongan_three.mat')
        # print(img)
        img = img['xiongan_three']
        gt = open_file(folder + 'xiongan_three_gt_changed.mat')['xiongan_three_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "pear",
                        "Soybean",
                        "aspen",
                        "glassland"]
        ignored_labels = [0]
    elif dataset_name == 'xiongan_four':

        img = open_file(folder + 'xiongan_four.mat')
        # print(img)
        img = img['xiongan_four']
        gt = open_file(folder + 'xiongan_four_gt_changed.mat')['xiongan_four_gt_changed']
        rgb_bands = (120, 72, 36)
        label_values = ["Undefined",
                        "corn",
                        "pear",
                        "Soybean",
                        "aspen",
                        "vegetable field",
                        "glassland"]
        ignored_labels = [0]
    elif dataset_name == 'Washington DC':

        img = open_file(folder + 'dc.mat')

        img = img['dc']
        gt = open_file(folder + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
        rgb_bands = (60, 27, 17)
        label_values = ["Undefined",
                       " Strawberry",
                        "Cowpea",
                        "Soybean",
                        "Sorghum",
                        "Water spinach",
                        "Watermelon",
                        "Greens"
               ]
        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-HanChuan':

        img = open_file(folder + 'WHU_Hi_HanChuan.mat')

        img = img['WHU_Hi_HanChuan']
        gt = open_file(folder + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
        rgb_bands = (110, 70, 33)
        label_values = ["Undefined",
                       " Strawberry",
                        "Cowpea",
                        "Soybean",
                        "Sorghum",
                        "Water spinach",
                        "Watermelon",
                        "Greens",
                        "Trees",
                        "Grass",
                        "Red roof",
                        "Gray roof",
                        "Plastic",
                        "Bare soil",
                        "Road",
                        "Bright object",
                        "Water"]
        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-HongHu':

        img = open_file(folder + 'WHU_Hi_HongHu.mat')

        img = img['WHU_Hi_HongHu']
        gt = open_file(folder + 'WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
        rgb_bands = (110, 70, 33)
        label_values = ["Undefined",
                        "Red roof",
                        "Road",
                        "Bare soil",
                        "Cotton",
                        "Cotton firewood",
                        "Rape",
                        "Chinese cabbage",
                        "Pakchoi",
                        "Cabbage",
                        "Tuber mustard",
                        "Brassica parachinensis",
                        "Brassica chinensis",
                        "Small Brassica chinensis",
                        "Lactuca sativa",
                        "Celtuce",
                        "Film covered lettuce",
                        "Romaine lettuce",
                        "Carrot",
                       " White radish",
                        "Garlic sprout",
                        "Broad bean",
                        "Tree"
                        ]
        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-LongKou':

        img = open_file(folder + 'WHU_Hi_LongKou.mat')

        img = img['WHU_Hi_LongKou']
        gt = open_file(folder + 'WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        rgb_bands = (110, 70, 33)
        label_values = ["Undefined",
                        "Corn",
                        "Cotton",
                        "Sesame",
                        "Broad-leaf soybean",
                        "Narrow-leaf soybean",
                        "Rice",
                        "Water",
                        "Roads and houses",
                        "Mixed weed"]
        ignored_labels = [0]
    elif dataset_name == 'NC12':
        data = hdf5storage.loadmat('/home/t1/DL_Code/DATASET/24-Yellow River-康旭东/NC12.mat')
        
        img = data['HSI']
        gt = data['GT']
        rgb_bands = (110, 70, 33)#?
        label_values = ["Undefined",
                        "Suaeda glauca",
                        "Reed",
                        "Water",
                        "White cloth",
                        "Withered reed",
                        "Tamarix",
                        "Stone",
                        "Moist soil",
                        "Mudflat",
                        "Spartina alterniflora",
                        "Dry oil",
                        "Standard reflectance cloth"]
        ignored_labels = [0]
    elif dataset_name == 'NC13':
        data = hdf5storage.loadmat('/home/t1/DL_Code/DATASET/24-Yellow River-康旭东/NC13.mat')
        
        img = data['HSI']
        gt = data['GT']
        rgb_bands = (110, 70, 33)#?
        label_values = [
            "Suaeda glauca",
            "Mixed asphalt cement road",
            "Moist soil",
            "Water",
            "Oil",
            "Reed",
            "Tamarix",
            "Car",
            "Dry soil",
            "Standard reflectance cloth",
            "Mixed tamarix reed",
            "Mixed suaeda glauca reed",
            "Mixed water reed"
        ]

        ignored_labels = [0]
    elif dataset_name == 'NC16':
        data = hdf5storage.loadmat('/home/t1/DL_Code/DATASET/24-Yellow River-康旭东/NC16.mat')
        
        img = data['HSI']
        gt = data['GT']
        rgb_bands = (110, 70, 33)#?
        label_values = [
            "Suaeda glauca",
            "Cement road",
            "Asphalt road",
            "Water",
            "Stone",
            "Hay",
            "Iron",
            "Tamarix",
            "Withered reed",
            "Reed",
            "Spartina alterniflora",
            "Moss",
            "Dry soil",
            "Moist soil",
            "Mudflat",
            "Standard reflectance cloth"
        ]

        ignored_labels = [0]
    else:
        # Custom dataset
        print("No dataset founded!")
        quit()
    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))

    if np.count_nonzero(nan_mask) > 0:
        print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img_original = img
    img_original = np.asarray(img_original, dtype='float32')


    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    if need_pca:
        # PCA transform
        from sklearn.decomposition import PCA
        num_components = 30
        new_data = np.reshape(img, (-1, img.shape[2]))
    
        # 1. PCA 降维
        pca = PCA(n_components=num_components, whiten=True)
        new_data = pca.fit_transform(new_data)
        
        # 2. 恢复形状
        img = np.reshape(new_data, (img.shape[0], img.shape[1], num_components))
        
    # Normalization
    img = np.asarray(img, dtype='float32')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # print(img[100, 100, :])
    return img, img_original, gt, label_values, ignored_labels, rgb_bands,palette


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.model = hyperparams['model']
        #print(self.model)
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        #self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = 1
        #supervision = hyperparams['supervision']
        supervision = 'full'
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            # link iterators
            mask1_each_class_number = Counter(list(itertools.chain.from_iterable(mask)))
            # print('when run def HyperX,生成一个跟输入 gt 一样大小的全 1 的矩阵，矩阵的元素个数是：', mask1_each_class_number)

            for l in self.ignored_labels:
                mask[gt == l] = 0
            mask2_each_class_number = Counter(list(itertools.chain.from_iterable(mask)))
            # print('when run def HyperX, 将是 ignored_label 的位置置 0 ，其他位置还是 1 ，各元素的个数是： ', mask2_each_class_number)
    
        # Semi-supervised : use all pixels, except padding #   except 除……之外
        elif supervision == 'semi':
            mask = np.ones_like(gt)

        x_pos, y_pos = np.nonzero(mask)  #返回数组 mask 中非零元素的索引值数组 tuple

        p = self.patch_size // 2
        # print('when run def HyperX,patch_size // 2 :', p)

        # self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
        #                          x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        #transform to list and to numpy array
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        print(len(self.indices), len(x_pos))

        self.labels = [self.label[x, y] for x, y in self.indices]
        #shuffle the data consequence
        # 
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        #flip with a certain probability
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    def __len__(self):
        # print('when run def __len__(self): the len(self.indices) : ',len(self.indices))
        return len(self.indices)

    def __getitem__(self, i):
        # print('def __getitem__(self, i): 中 i ：',i)
        length, width, n_band = self.data.shape[0], self.data.shape[1], self.data.shape[2]
        # 在 生成 数据块
        # print(length)
        x, y = self.indices[i]
        # x, y = 268,66
        # print(x,y)
        # quit()
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2,:]
        label = self.label[x1:x2, y1:y2]

        if self.model in ["dbda"]:
            pass
        else:
            data = np.asarray(np.copy(data), dtype='float32')
     
        label = np.asarray(np.copy(label), dtype='int64')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # print('when run def HyperX,改变data形状之前的data.size():',data.size())
        label = torch.from_numpy(label)
        # print('when run def HyperX,改变label形状之前的label.size():',label.size())

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            
            label = label[self.patch_size // 2, self.patch_size // 2]

            # label = torch.tensor([label, x, y])

        return data, label
    

class HyperY(torch.utils.data.Dataset):

  
    def __init__(self, data, gt,  **hyperparams):

        super(HyperY, self).__init__()

        self.data = data
        self.label = gt
        
 
    def __len__(self):

        return len(self.label)
 
    def __getitem__(self, i):
        
        feature = self.data[i]

        label = self.label[i]

       
        return feature, label
    
class HyperZ(torch.utils.data.Dataset):

  
    def __init__(self, data, gt, coor_x, coor_y, **hyperparams):

        super(HyperZ, self).__init__()

        self.data = data
        self.label = gt
        self.pos_x = coor_x
        self.pos_y = coor_y
 
    def __len__(self):

        return len(self.label)
 
    def __getitem__(self, i):
        
        feature = self.data[i]

        label = self.label[i]

        coor_x = self.pos_x[i]
        coor_y = self.pos_y[i]

        return feature, label, coor_x, coor_y
    
#test the marked label    
class Feature_Distribute(torch.utils.data.Dataset):

  
    def __init__(self, feature, gt,  **hyperparams):

        super(Feature_Distribute, self).__init__()

        self.feature = feature
        self.label = gt

        x_coor, y_coor = np.nonzero(gt) #test all index

        self.indices = np.array([(x, y) for x, y in zip(x_coor, y_coor)])
        np.random.shuffle(self.indices)

    def __len__(self):

        return len(self.indices)
 
    def __getitem__(self, i):

        x_pos = self.indices[i][0]
        y_pos = self.indices[i][1]
        
        feature = self.feature[x_pos, y_pos]

        label = self.label[x_pos, y_pos]

       
        return feature, label, self.indices[i]

# test all the labels    
class Feature_Distribute1(torch.utils.data.Dataset):

  
    def __init__(self, feature, gt,  **hyperparams):

        super(Feature_Distribute1, self).__init__()

        self.feature = feature
        self.label = gt

        x_coor, y_coor = np.where(gt >= 0)
        self.indices = np.array([(x, y) for x, y in zip(x_coor, y_coor)])
        np.random.shuffle(self.indices)

    def __len__(self):

        return len(self.indices)
 
    def __getitem__(self, i):

        x_pos = self.indices[i][0]
        y_pos = self.indices[i][1]
        
        feature = self.feature[x_pos, y_pos]

        label = self.label[x_pos, y_pos]

       
        return feature, label, self.indices[i]
    

class Patch_Dataset(torch.utils.data.Dataset):

    def __init__(self, img, patch_size, gt):
        
        self.patch_size = patch_size
        self.gt = gt

        x_coor, y_coor = np.where(gt > 0)
        self.indices = np.array([(x, y) for x, y in zip(x_coor, y_coor)])
        np.random.shuffle(self.indices)

        self.data = img
        self.labels = []
        self.patches = []
        for x_pos, y_pos in self.indices:
            patch = self.data[x_pos:x_pos + self.patch_size, y_pos: 
                         y_pos + self.patch_size, :]
            self.patches.append(patch)
            self.labels.append(gt[x_pos, y_pos])

    def __len__(self):

        return len(self.indices)
    
    def __getitem__(self, i):

        return self.patches[i], self.labels[i], self.indices[i]

class Patch_Dataset1(torch.utils.data.Dataset):

    def __init__(self, img, patch_size, gt):
        
        self.patch_size = patch_size
        self.gt = gt

        x_coor, y_coor = np.where(gt >= 0)
        self.indices = np.array([(x, y) for x, y in zip(x_coor, y_coor)])
        np.random.shuffle(self.indices)

        self.data = img
        self.labels = []
        self.patches = []
        for x_pos, y_pos in self.indices:
            patch = self.data[x_pos:x_pos + self.patch_size, y_pos: 
                         y_pos + self.patch_size, :]
            self.patches.append(patch)
            self.labels.append(gt[x_pos, y_pos])

    def __len__(self):

        return len(self.indices)
    
    def __getitem__(self, i):

        return self.patches[i], self.labels[i], self.indices[i]
