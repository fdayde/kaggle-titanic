import os


class PathManager:
    """
    Manages dataset and working paths based on the execution environment (Kaggle or local).

    Attributes:
        dataset_path (str): The path to the dataset directory.
        working_path (str): The path to the working directory.
    """

    def __init__(self, dataset_name: str = "titanic"):
        """
        Initializes PathManager by setting and validating dataset and working paths.

        Args:
            dataset_name (str, optional): The name of the dataset on Kaggle. Defaults to "titanic".
        """
        self.dataset_name = dataset_name
        self.dataset_path: str
        self.working_path: str
        self.set_paths()

    def set_paths(self) -> None:
        """
        Sets the dataset and working paths based on the environment (Kaggle or local).

        Raises:
            FileNotFoundError: If the dataset or working paths do not exist.
        """
        # Determine environment and set paths
        if os.path.exists('/kaggle'):
            print("Running on Kaggle.")
            self.dataset_path = '/kaggle/input/{self.dataset_name}'
            self.working_path = '/kaggle/working'
        else:
            print("Running on local.")
            # Assuming the notebook is in the 'notebooks' folder
            project_root = os.path.abspath('../')
            data_dir = os.path.join(project_root, 'data')
            working_dir = os.path.join(data_dir, 'working')

            print(f"Dataset directory: {data_dir}")
            print(f"Working directory: {working_dir}")

            self.dataset_path = data_dir
            self.working_path = working_dir

        # Validate paths
        self.validate_paths()

    def validate_paths(self) -> None:
        """
        Validates that the dataset and working paths exist.

        Raises:
            FileNotFoundError: If the dataset or working paths do not exist.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"The dataset path does not exist: {self.dataset_path}")

        if not os.path.exists(self.working_path):
            raise FileNotFoundError(f"The working path does not exist: {self.working_path}")

    def __str__(self) -> str:
        """
        Returns a formatted string representation of the dataset and working paths.

        Returns:
            str: Formatted dataset and working paths.
        """
        return (
            f"Dataset Path: {self.dataset_path}\n"
            f"Working Path: {self.working_path}"
        )