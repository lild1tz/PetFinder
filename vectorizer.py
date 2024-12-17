import torch # type: ignore
from torchvision import transforms # type: ignore
from PIL import Image # type: ignore
from roboflow import Roboflow # type: ignore
from simclr import SimCLR  # type: ignore
from collections import OrderedDict


class Vectorizer:
    def __init__(self, params, weight_path_resnet14, weight_path_simclr, roboflow_api_key, workspace, project_name, version, device="cpu"):
        """
        Инициализация класса Vectorizer для детекции и векторизации изображений.

        :param params: Параметры модели SimCLR.
        :param weight_path_resnet14: Путь к весам ResNet14.
        :param weight_path_simclr: Путь к весам SimCLR.
        :param roboflow_api_key: Ключ Roboflow API.
        :param workspace: Название рабочей области в Roboflow.
        :param project_name: Название проекта в Roboflow.
        :param version: Версия модели Roboflow.
        :param device: Устройство для выполнения вычислений ("cpu" или "cuda").
        """
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Загрузка модели SimCLR
        self.model = self._load_model(params, weight_path_resnet14, weight_path_simclr)
        
        # Инициализация детектора Roboflow
        self.roboflow_model = self._load_roboflow_model(roboflow_api_key, workspace, project_name, version)
        
        # Трансформации для изображения
        self.transform = self._get_transform()

    def _load_model(self, params, weight_path_resnet14, weight_path_simclr):
        """Загрузка модели SimCLR с весами."""
        model = SimCLR(params, pretrained_weights_path=weight_path_resnet14).to(self.device)
        
        simclr_state_dict = torch.load(weight_path_simclr, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in simclr_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model

    def _load_roboflow_model(self, api_key, workspace, project_name, version):
        """Инициализация модели детекции Roboflow."""
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        return project.version(version).model

    def _get_transform(self):
        """Возвращает трансформации для изображения."""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def detect_objects(self, image_path, confidence=40, overlap=30):
        """
        Выполняет детекцию объектов на изображении с помощью Roboflow.

        :param image_path: Путь к изображению.
        :param confidence: Порог уверенности детекции.
        :param overlap: Допустимое перекрытие.
        :return: Список предсказанных объектов с их координатами.
        """
        result = self.roboflow_model.predict(image_path, confidence=confidence, overlap=overlap).json()
        return result#.get("predictions", [])

    def vectorize_cropped_objects(self, image_path, detections):
        """
        Векторизует обрезанные объекты на изображении.

        :param image_path: Путь к изображению.
        :param detections: Список объектов с координатами.
        :return: Список векторов для каждого объекта.
        """
        image = Image.open(image_path).convert("RGB")
        vectors = []

        for i, det in enumerate(detections):
            # Обрезаем изображение по координатам
            x, y, w, h = det["x"], det["y"], det["width"], det["height"]
            cropped_image = image.crop((x - w // 2, y - h // 2, x + w // 2, y + h // 2))

            # Применяем трансформации и векторизуем
            image_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vector = self.model(image_tensor).squeeze().cpu()
            vectors.append({"object_class": det["class"], "vector": vector})
        
        return vectors

    def process_image(self, image_path):
        """
        Объединяет детекцию объектов и их векторизацию.

        :param image_path: Путь к изображению.
        :return: Список векторов и классов объектов.
        """
        # Шаг 1: Детекция объектов
        detections = self.detect_objects(image_path)
        if not detections:
            print("No objects detected.")
            return []

        # Шаг 2: Векторизация обрезанных объектов
        vectors = self.vectorize_cropped_objects(image_path, detections)
        return vectors


