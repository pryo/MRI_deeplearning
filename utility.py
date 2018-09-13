
import matplotlib.pyplot as plt
from dataset import *
import torchvision.transforms as transforms
import densenet as MD
import pickle
import torch.nn as nn
import torch.nn as nn
def readCoor(ID, df):

    return eval(df.loc[int(ID)][1])
def getPath(root_dir,patientID):
    dir = os.path.join(root_dir,
                       patientID)
    apt_img_path = None
    for name in os.listdir(dir):
        if name.endswith('.img') and 'aptw' in name:
            apt_img_path = os.path.join(dir, name)
    return apt_img_path
def readImgArray(path):
    dtype = np.dtype('float32')
    fid = open(path, 'rb')
    data = np.fromfile(fid, dtype)
    try:
        matrix = data.reshape((400, 400))
        matrix = np.flipud(matrix)
        fid.close()
    except ValueError:
        print(str(id) + 'is not 400 by 400, skipped to next...')
        fid.close()
    return matrix
def array2CVImg(array):
    #image = None
    image = np.clip(array,-5,5)+5

    image = (image/10)*255
    #image = np.interp(image, (image.min(), image.max()), (0, 255))
    return image
def getSample(ID):
    log_dir = r"C:\Users\wzuo\Developer\ML for APT\APT\1D_APT_ROI_Log.csv"
    dataframe = pd.read_csv(log_dir, header=None, index_col=0)

    coor = readCoor(ID,dataframe)
    # coor 0:x1 1:y1 2:x2 3:y2
    coor = [round(x) for x in coor]
    data_dir = r"C:\Users\wzuo\Developer\ML for APT\data"
    path = getPath(data_dir, ID)
    m = readImgArray(path)
    image = array2CVImg(m)
    patch = image[coor[1]:coor[3], coor[0]:coor[2]]
    return patch, image


def showSample(ID):
    ID  = str(ID)
    patch,image = getSample(ID)
    fig = plt.figure(figsize=(8, 6), dpi=80)
    whole = fig.add_subplot(2, 1, 1)
    whole.imshow(image)
    local = fig.add_subplot(2, 1, 2)
    local.imshow(patch)
    plt.show()


def getLoader():
    data_dir = r"C:\Users\wzuo\Developer\ML for APT\data"
    gz_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\gz_list.p"
    kki_list_path = r"C:\Users\wzuo\Developer\ML for APT\data\kki_list.p"
    ROI_log_path = r"C:\Users\wzuo\Developer\ML for APT\APT\1D_APT_ROI_Log.csv"
    gz_list=pickle.load( open( gz_list_path, "rb" ) )
    kki_list=pickle.load( open( kki_list_path, "rb" ) )
    img_transform = transforms.Compose(
        [transforms.Resize([224,224]),
            transforms.ToTensor()
            ])
    dataset_switch=1
    apt_dataset = DualChannelAPTDataset(data_dir,gz_list,kki_list,ROI_log_path,
                             truth_path=r"C:\Users\wzuo\Developer\ML for APT\idh.xlsx",
                             transform=img_transform,switch=dataset_switch)
    model_path = r'C:\Users\wzuo\Developer\ML for APT\models\1534191799.996107.model'
    patch_model_path=r'C:\Users\wzuo\Developer\ML for APT\models\1534191799.996107.patchModel'
    loader = torch.utils.data.DataLoader(apt_dataset,
                    batch_size=1)
    return loader
def saveToDisk(ID,path):
    #baddiePath = r"C:\Users\wzuo\Developer\ML for APT\APT\baddie"
    ID = str(ID)
    patch, image = getSample(ID)
    fig = plt.figure(figsize=(8, 6), dpi=80)
    whole = fig.add_subplot(2, 1, 1)
    whole.imshow(image)
    local = fig.add_subplot(2, 1, 2)
    local.imshow(patch)
    filePath =  os.path.join(path,
                       ID+'.png')
    fig.savefig(filePath)
    plt.close()
#
# def getRGBimg(id,PIL=True):
#     if PIL:
#

def findBadAppleinDualDenseExtClassifier(extClassiferPath,loader):

    # modelPath = r'C:\Users\wzuo\Developer\ML for APT\models\1534191799.996107.model'
    # patchModelPath=r'C:\Users\wzuo\Developer\ML for APT\models\1534191799.996107.patchModel'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wrongs=[]
    #model = MD.densenet201()
    model_global = MD.densenet201(pretrained=True)
    model_local = MD.densenet201(pretrained=True)
    # todo change model init
    #patchModel =MD.SimpleNet()
    # todo change second model init
    #num_ftrs = model.classifier.in_features


    num_ftrs_global = model_global.classifier.in_features

    num_ftrs_local = model_local.classifier.in_features

    externalClassifier = nn.Sequential(
        nn.Linear(num_ftrs_global + num_ftrs_local + 1, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    # externalClassifier = nn.Sequential(
    #     nn.Linear(num_ftrs_global + num_ftrs_local + 1, 2)
    # )

    # todo init the external classifier instead of the whole two densenet model

    #the_model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(modelPath))
    externalClassifier.load_state_dict(torch.load(extClassiferPath))
    # todo load the params in the external classifier
    #patchModel.load_state_dict(torch.load(patchModelPath))
    #model = model.to(device)
    #patchModel = patchModel.to(device)
    model_global = model_global.to(device)
    model_local = model_local.to(device)
    externalClassifier = externalClassifier.to(device)
    # todo send the external classifer to the device

    #patchModel.eval()
    #model.eval()
    model_local.eval()
    model_global.eval()
    externalClassifier.eval()
    # set all separate model to eval()

    for batch_idx,sample in enumerate(loader):
        images = sample['image'].to(device)
        patchs = sample['patch'].to(device)
        ages = sample['age'].to(device)
        labels = sample['label'].to(device)
        ids = sample['id']
        outputLocal = model_local(patchs)
        outputGlobal = model_global(images)
        interim = torch.cat((outputGlobal, outputLocal), 1)
        interim = torch.cat((ages.unsqueeze(1), interim), 1)
        outputs = externalClassifier(interim)
        _, preds = torch.max(outputs, 1)
        #wrongs.append(ids[preds != labels.data])
        #print(ids)
        #print(preds != labels.data)
        if (preds != labels.data)[0]:
            wrongs.append(ids[0])
            # save the entire sample
    return wrongs