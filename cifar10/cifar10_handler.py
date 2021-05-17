"""
Module for image classification default handler
"""
"""
Module for image classification default handler
"""
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
import torch


"""
Base module for all vision handlers
"""
from abc import ABC
import io
import os
import base64
import torch
import numpy as np
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from ts.torch_handler.base_handler import BaseHandler
from classifier import CIFAR10CLASSIFIER
import logging
from captum.attr import Saliency
import os
from torchvision import transforms
import torch
logger = logging.getLogger(__name__)

class CIFAR10Classification(BaseHandler, ABC):
    """
    Base class for all vision handlers
    """
    def initialize(self, ctx):
        """In this initialize function, the Titanic trained model is loaded and
        the Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        print("Model dir is {}".format(model_dir))
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )

        self.model = CIFAR10CLASSIFIER()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()

        logger.info("CIFAR10 model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        # mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # if os.path.isfile(mapping_file_path):
        #     print("Mapping file present")
        #     with open(mapping_file_path) as f:
        #         self.mapping = json.load(f)
        # else:
        #     print("Mapping file missing")
        #     logger.warning("Missing the index_to_name.json file.")


        self.ig = IntegratedGradients(self.model)
        self.nt = NoiseTunnel(self.ig)
        self.dl = DeepLift(self.model)
        self.initialized = True
        self.saliency = Saliency(self.model)
        topk = 5
        self.image_processing = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.
        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionary with predictons and explanations are returned.
        """
        return data.argmax(1).tolist()

    def attribute_image_features(self,algorithm, data, **kwargs):
        self.model.zero_grad()
        tensor_attributions = algorithm.attribute(data,
                                              target=0,
                                              **kwargs
                                             )
        return tensor_attributions

    def get_insights(self, tensor_data, _, target=0):
        explanation_dict = {}
        attr_ig, delta = self.attribute_image_features(self.ig, tensor_data, baselines=tensor_data * 0, return_convergence_delta=True,n_steps=200)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        # # attr_ig_nt = self.attribute_image_features(self.nt, tensor_data, baselines=tensor_data * 0, nt_type='smoothgrad_sq',
        # #                               nt_samples=100, stdevs=0.2)
        # # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # # grads = self.saliency.attribute(tensor_data, target=0)
        # # grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        # attr_dl = self.attribute_image_features(self.dl, tensor_data, baselines=tensor_data * 0)
        # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # explanation_dict["Integrated_Gradients"]=attr_ig.tolist()
        # explanation_dict["saliency"]=grads.tolist()
        # explanation_dict["DeepLift"]=attr_dl.tolist()

        return attr_ig.tolist()

        