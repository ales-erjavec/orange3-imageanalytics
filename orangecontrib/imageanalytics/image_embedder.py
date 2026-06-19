from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from Orange.data import ContinuousVariable, Domain, Table, Variable
from Orange.data.util import SharedComputeValue
from Orange.misc.utils.embedder_utils import EmbedderCache
from Orange.util import dummy_callback
from orangecontrib.imageanalytics.timm_model import InceptionV3, \
    InceptionV4, ResNet18, ResNet50, ConvNeXt_Small, ConvNeXt_Tiny, ConvNeXt_Atto, \
    Inception_Next_Atto, Inception_Next_Tiny

from orangecontrib.imageanalytics.local_embedder import LocalEmbedder
from orangecontrib.imageanalytics.server_embedder import ServerEmbedder
from orangecontrib.imageanalytics.squeezenet_model import SqueezenetModel
from orangecontrib.imageanalytics.utils.image_utils import extract_paths

MODELS = {
    "inceptionnext_atto-local": {
        "name": "InceptionNeXt Atto",
        "description": "InceptionNeXt image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 0,
        "is_local": True,
        "model": Inception_Next_Atto,
    },
    "inceptionnext_tiny-local": {
        "name": "InceptionNeXt Tiny",
        "description": "InceptionNeXt image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 0,
        "is_local": True,
        "model": Inception_Next_Tiny,
    },
    "inception-v3-local": {
        "name": "Inception v3",
        "description": "Google's Inception v3 model trained on ImageNet.",
        "target_image_size": (299, 299),
        "layers": ["penultimate"],
        "order": 0,
        "is_local": True,
        "model": InceptionV3,
    },
    "inception-v3": {
        "name": "Inception v3",
        "description": "Google's Inception v3 model trained on ImageNet.",
        "target_image_size": (299, 299),
        "layers": ["penultimate"],
        "order": 1,
        # batch size tell how many images we send in parallel, this number is
        # high for inception since it has many workers, but other embedders
        # send less images since bottleneck are workers, this way we avoid
        # ReadTimeout because of images waiting in a queue at the server
        "batch_size": 500,
    },
    "inception-v4-local": {
        "name": "Inception v4",
        "description": "Google's Inception v4 model trained on ImageNet.",
        "target_image_size": (299, 299),
        "layers": ["penultimate"],
        "order": 1,
        "is_local": True,
        "model": InceptionV4,
    },
    "resnet18-local": {
        "name": "ResNet-B (18)",
        "description": "18-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "is_local": True,
        "model": ResNet18,
    },
    "resnet50-local": {
        "name": "ResNet-B (50)",
        "description": "50-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "is_local": True,
        "model": ResNet50,
    },
    "convnext_small-local": {
        "name": "ConvNeXt Small",
        "description": "ConvNeXt image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "is_local": True,
        "model": ConvNeXt_Small,
    },
    "convnext_tiny-local": {
        "name": "ConvNeXt Tiny",
        "description": "ConvNeXt image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "is_local": True,
        "model": ConvNeXt_Tiny,
    },
    "convnext_atto-local": {
        "name": "ConvNeXt Atto",
        "description": "ConvNeXt image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "is_local": True,
        "model": ConvNeXt_Atto,
    },
    "painters": {
        "name": "Painters",
        "description": "A model trained to predict painters from artwork\nimages.",
        "target_image_size": (256, 256),
        "layers": ["penultimate"],
        "order": 4,
        "batch_size": 500,
    },
    "deeploc": {
        "name": "DeepLoc",
        "description": "A model trained to analyze yeast cell images.",
        "target_image_size": (64, 64),
        "layers": ["penultimate"],
        "order": 5,
        "batch_size": 500,
    },
    "vgg16": {
        "name": "VGG-16",
        "description": "16-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 2,
        "batch_size": 500,
    },
    "vgg19": {
        "name": "VGG-19",
        "description": "19-layer image recognition model trained on\nImageNet.",
        "target_image_size": (224, 224),
        "layers": ["penultimate"],
        "order": 3,
        "batch_size": 500,
    },
    "openface": {
        "name": "openface",
        "description": "Face recognition model trained on FaceScrub and\n"
        "CASIA-WebFace datasets.",
        "target_image_size": (256, 256),
        "layers": ["penultimate"],
        "order": 6,
        "batch_size": 500,
    },
    "squeezenet": {
        "name": "SqueezeNet",
        "description": "Deep model for image recognition that achieves \n"
        "AlexNet-level accuracy on ImageNet with \n"
        "50x fewer parameters.",
        "target_image_size": (227, 227),
        "layers": ["penultimate"],
        "order": 1,
        "is_local": True,
        "batch_size": 16,
        "model": SqueezenetModel,
    },
}


class ImageEmbedder:
    """
    Client side functionality for accessing a remote image embedding backend.

    Attributes
    ----------
    model
        Name of the model, must be one from MODELS dictionary
    server_url
        The url of the server with embedding backend.

    Examples
    --------
    >>> import Orange
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

    >>> # embedding from list of paths
    >>> image_file_paths = ['image001.jpg', 'image001.jpg']
    >>> with ImageEmbedder(model='model_name') as emb:
    ...    embeddings = emb(image_file_paths)

    >>> # embedding from orange tabl
    >>> table = Orange.data.Table('Table_with_image_path.csv')
    >>> with ImageEmbedder(model='model_name') as emb:
    ...    embeddings = emb(table, col="image_path_column")
    """

    _embedder = None

    def __init__(
        self,
        model: str = "inception-v3",
        server_url: str = "https://api.garaza.io/",
    ):
        self.server_url = server_url
        self.model = model
        self._model_settings = self._get_model_settings_confidently()

    def is_local_embedder(self) -> bool:
        """
        Tells whether selected embedder is local or not.
        """
        return self._model_settings.get("is_local", False)

    def _get_model_settings_confidently(self) -> Dict[str, Any]:
        """
        Returns the dictionary with model settings

        Returns
        -------
        The dictionary with model settings
        """
        if self.model not in MODELS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ", ".join(MODELS.keys())
            raise ValueError(model_error.format(self.model, available_models))
        return MODELS[self.model]

    def _init_embedder(self) -> None:
        """
        Init local or server embedder.
        """
        if self.is_local_embedder():
            bsize = self._model_settings.get("batch_size", LocalEmbedder.DEFAULT_BATCH_SIZE)
            self._embedder = LocalEmbedder(self.model, self._model_settings, batch_size=bsize)
        else:
            self._embedder = ServerEmbedder(
                self.model,
                self._model_settings["batch_size"],
                self.server_url,
                "image",
                self._model_settings["target_image_size"]
            )

    def __call__(
        self,
        data: Union[Table, List[str], np.array],
        col: Optional[Union[str, Variable]] = None,
        callback: Optional[Callable] = dummy_callback,
        enable_domain_transform: bool = False,
    ) -> Union[Tuple[Table, Table, int], List[List[float]]]:
        """
        Embedd images.

        Parameters
        ----------
        data
            Data contains the path to images (locally or online). It can be
            Orange data table or list/array. When data table on input col
            parameter must define which column in the table contains images.
        col
            The column with images in Orange data table. It is not required
            when data are list or array.
        callback
            Optional callback - function that is called for every embedded
            image and is used to report the progress.
        enable_domain_transform: bool
            If `True` record the transform in the resulting tables domain
            (see `Variable.compute_value`)

        Returns
        -------
        Embedded images. When data is Table it returns tuple with two tables:
        1) original table with embedded images appended to it, 2) table with
        skipped images, 3) number of skipped images.
        When data is array/list it returns the list of list with embeddings,
        each image is represented with vector of numbers.
        """
        assert data is not None
        assert isinstance(data, (np.ndarray, list, Table))
        self._init_embedder()
        if isinstance(data, Table):
            assert col is not None, "Please provide a column for image path"
            # if table on input tables on output
            return self.from_table(data, col=col, callback=callback,
                                   enable_domain_transform=enable_domain_transform)
        elif isinstance(data, (np.ndarray, list)):
            # if array-like on input array-like on output
            return self._embedder.embedd_data(data, callback=callback)

    def from_table(
        self,
        data: Table,
        col: Union[str, Variable] = "image",
        callback: Callable = None,
        enable_domain_transform=False,
    ) -> Tuple[Table, Table, int]:
        """
        Calls embedding when data are provided as a Orange Table.

        Parameters
        ----------
        data
            Data table with image paths
        col
            The column with image paths
        callback
            Optional callback - function that is called for every embedded
            image and is used to report the progress.
        enable_domain_transform: bool
            If `True` record the transform in the resulting tables domain
            (see `Variable.compute_value`)
        """
        file_paths = extract_paths(data, data.domain[col])
        embeddings = self._embedder.embedd_data(file_paths, callback=callback)
        shared_compute = None
        dims = max((len(e) for e in embeddings if e is not None), default=0)
        if enable_domain_transform and dims:
            shared_compute = ImageEmbedderTransform.Embedder(
                self.model, data.domain[col].name, dims
            )
        return ImageEmbedder.prepare_output_data(data, embeddings, shared_compute=shared_compute)

    def __enter__(self) -> "ImageEmbedder":
        return self

    def __exit__(self, _, __, ___) -> None:
        pass

    def __del__(self) -> None:
        self.__exit__(None, None, None)

    @staticmethod
    def construct_output_data_table(
            embedded_images: Table, embeddings_: np.ndarray,
            shared_compute: Optional['ImageEmbedderTransform.Embedder'] = None
    ) -> Table:
        """
        Join the orange table with embeddings.

        Parameters
        ----------
        embedded_images
            Table with images that were successfully embedded
        embeddings_
            Embeddings for images from table
        shared_compute

        Returns
        -------
        Table with added embeddings to data.
        """
        new_attributes = [
            ContinuousVariable(
                "n{:d}".format(i),
                compute_value=ImageEmbedderTransform(shared_compute, i) if shared_compute else None
            )
            for i in range(embeddings_.shape[1])
        ]

        # prevent embeddings to be shown in long drop-downs in e.g. scatterplot
        for a in new_attributes:
            a.attributes["hidden"] = True

        embeddings_table = embedded_images.from_numpy(Domain(new_attributes), embeddings_)
        return embedded_images.concatenate([embedded_images, embeddings_table], axis=1)

    @staticmethod
    def prepare_output_data(
            input_data: Table,
            embeddings_: List[List[float]],
            shared_compute: Optional['ImageEmbedderTransform.Embedder']=None
    ) -> Tuple[Table, Table, int]:
        """
        Prepare output data when data table on input.

        Parameters
        ----------
        input_data
            The table with original data that are joined with embeddings
        embeddings_
            List with embeddings
        shared_compute
        Returns
        -------
        Tuple where first parameter is table with embedded images, the second
        table with skipped images and third the number of skipped images.
        """
        skipped_images_bool = [x is None or len(x) == 0 for x in embeddings_]

        if np.any(skipped_images_bool):
            skipped_images = input_data[skipped_images_bool].copy()
            skipped_images.name = "Skipped images"
            num_skipped = len(skipped_images)
        else:
            num_skipped = 0
            skipped_images = None

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = input_data[embedded_images_bool]

            embeddings_ = [
                e for e, b in zip(embeddings_, embedded_images_bool) if b
            ]
            embeddings_ = np.vstack(embeddings_)

            embedded_images = ImageEmbedder.construct_output_data_table(
                embedded_images, embeddings_, shared_compute=shared_compute
            )
            embedded_images.ids = input_data.ids[embedded_images_bool]
            embedded_images.name = "Embedded images"
        else:
            embedded_images = None

        return embedded_images, skipped_images, num_skipped

    def clear_cache(self) -> None:
        """
        Function clear cache for the selected embedder. If embedder is loaded
        cache is cleaned from its dict otherwise we load cache and clean it
        from file.
        """
        if self._embedder:
            # embedder is loaded so we clean its cache
            self._embedder.clear_cache()
        else:
            # embedder is not initialized yet - clear it cache from file
            cache = EmbedderCache(self.model)
            cache.clear_cache()


class ImageEmbedderTransform(SharedComputeValue):
    class Embedder:
        def __init__(self, model: str, image_var_name: str, embedding_dim: int):
            self.model = model
            self.image_var_name = image_var_name
            self.embedding_dim = embedding_dim

        def __call__(self, table: Table):
            embedder = ImageEmbedder(self.model)
            paths = extract_paths(table, table.domain[self.image_var_name])
            emb = embedder(paths)
            # Replace empty results with nan
            nanv = [np.nan] * self.embedding_dim
            emb = [e if e is not None else nanv for e in emb]
            emb = np.array(emb)
            return emb

        def __hash__(self):
            return hash((self.model, self.image_var_name, self.embedding_dim))

        def __eq__(self, other):
            return (type(other) is ImageEmbedderTransform.Embedder
                    and self.model == other.model
                    and self.image_var_name == other.image_var_name
                    and self.embedding_dim == other.embedding_dim)

    def __init__(self, compute_shared: Embedder, index: int):
        super().__init__(compute_shared)
        self.index = index

    def compute(self, data: Table, shared_data: np.ndarray):
        return shared_data[:, self.index]

    def __hash__(self):
        return hash((super().__hash__(), self.index))

    def __eq__(self, other):
        return super().__eq__(other) and self.index == other.index


if __name__ == "__main__":
    image_file_paths = ["tests/test_images/example_image_0.jpg"]
    # with ImageEmbedder(model='inception-v3') as embedder:
    with ImageEmbedder(model="squeezenet") as embedder_:
        embedder_.clear_cache()
        print(embedder_(image_file_paths))
