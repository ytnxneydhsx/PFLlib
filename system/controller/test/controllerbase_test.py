import sys
import os


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


sys.path.insert(0, project_root)


from system.controller.controllerbase import controllerbase
import argparse
from system.flcore.trainmodel import models

parser = argparse.ArgumentParser()

args = parser.parse_args()
args.batch_size=50
args.dataset = 'Cifar10'
args.num_classes = 10
args.device='cuda'
args.model = models.VGG16_cifar10().to(args.device)
args.global_model = args.model


test=controllerbase(args)





