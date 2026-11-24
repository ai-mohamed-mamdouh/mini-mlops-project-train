import bentoml
import joblib
from bentoml.models import BentoModel
from pydantic import BaseModel, Field

from mini_mlops.config.loader import load_training_config

config = load_training_config()


class WineFeatures(BaseModel):
    alcohol: float = Field(gt=0)
    malic_acid: float = Field(gt=0)
    ash: float = Field(gt=0)
    alcalinity_of_ash: float = Field(gt=0)
    magnesium: float = Field(gt=0)
    total_phenols: float = Field(gt=0)
    flavanoids: float = Field(gt=0)
    nonflavanoid_phenols: float = Field(gt=0)
    proanthocyanins: float = Field(gt=0)
    color_intensity: float = Field(gt=0)
    hue: float = Field(gt=0)
    od280_od315_of_diluted_wines: float = Field(gt=0)
    proline: float = Field(gt=0)


image = bentoml.images.Image(python_version="3.11").python_packages(
    "scikit-learn", "joblib"
)


@bentoml.service(image=image, workers=1)
class WineService:
    model_ref = BentoModel(f"{config.model_name}:latest")

    def __init__(self):
        self.model = joblib.load(self.model_ref.path_of("model.pkl"))

    @bentoml.api
    def predict(self, sample: WineFeatures) -> dict:

        features = [
            [
                sample.alcohol,
                sample.malic_acid,
                sample.ash,
                sample.alcalinity_of_ash,
                sample.magnesium,
                sample.total_phenols,
                sample.flavanoids,
                sample.nonflavanoid_phenols,
                sample.proanthocyanins,
                sample.color_intensity,
                sample.hue,
                sample.od280_od315_of_diluted_wines,
                sample.proline,
            ]
        ]

        prediction = self.model.predict(features)

        class_id = int(prediction[0])

        return {"class_id": class_id}
