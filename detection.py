from roboflow import Roboflow  # type: ignore

class ObjectDetector:
    def __init__(self, api_key, workspace_name, project_name, version):
        """
        Инициализация детектора объектов с использованием Roboflow API.
        
        :param api_key: Ключ доступа к Roboflow API.
        :param workspace_name: Название рабочей области.
        :param project_name: Название проекта.
        :param version: Версия модели.
        """
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.project_name = project_name
        self.version = version
        self.model = self._load_model()

    def _load_model(self):
        """Загрузка модели из Roboflow."""
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace_name).project(self.project_name)
        return project.version(self.version).model

    def predict(self, image_path, confidence=40, overlap=30):
        """
        Выполняет предсказание для изображения.
        
        :param image_path: Путь до изображения.
        :param confidence: Порог уверенности (от 0 до 100).
        :param overlap: Допустимое перекрытие объектов (от 0 до 100).
        :return: Результат предсказания в формате JSON.
        """
        result = self.model.predict(image_path, confidence=confidence, overlap=overlap)
        return result.json()
