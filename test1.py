import os
import cv2
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util.visualizer import save_images2
from util import html
import util
from PIL import Image
import torchvision.transforms as transforms


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    #data_loader = CreateDataLoader(opt)
    #dataset = data_loader.load_data()

    model = create_model(opt)
    model.setup(opt)

    i = 0
    # test
    while True:
        images = []
        for root, _, fnames in sorted(os.walk("datasets/test1/")):
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)

        A = Image.open(images[0]).convert('RGB')
        A = transforms.ToTensor()(A)
        #A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        A = A.unsqueeze(0) 
        model.set_input(A)
        model.test()
        fake = util.tensor2im(model.fake_B.data)
        print(fake)
        assert False
        visuals = model.get_current_visuals()
        i = i + 1
        fileDelPath=images[0]
        fileDelName=os.path.basename(fileDelPath)
        print(fileDelName)
        src=cv2.imread(fileDelPath)
        cv2.namedWindow("test",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("test",src)
        os.remove(fileDelPath)
        cv2.waitKey(50)
       


