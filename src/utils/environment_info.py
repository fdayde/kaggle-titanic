import platform

import psutil
import torch


class EnvironmentInfo:
    """Class to retrieve and represent system environment information, including CPU and GPU details."""
    def __init__(self):
        """Initialize EnvironmentInfo by gathering GPU and CPU information."""
        self.gpu_info = self.get_gpu_info()
        self.cpu_info = self.get_cpu_info()
        self.cpu_details = self.get_cpu_details()
    
    def get_gpu_info(self) -> str:
        """Retrieve GPU information if available."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_details = [
                f"GPU {i} - Name: {torch.cuda.get_device_name(i)}, "
                f"Capability: {torch.cuda.get_device_capability(i)}"
                for i in range(num_gpus)
            ]
            return f"Number of GPUs: {num_gpus}\n" + "\n".join(gpu_details)
        else:
            return "No GPU available"
    
    def get_cpu_info(self) -> str:
        """Retrieve basic CPU information."""
        return platform.processor()
    
    def get_cpu_details(self) -> str:
        """Retrieve detailed CPU information."""
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        return (
            f"Physical cores: {cpu_count}\n"
            f"Total cores: {cpu_count_logical}\n"
            f"Max Frequency: {cpu_freq.max:.2f}MHz\n"
            f"Min Frequency: {cpu_freq.min:.2f}MHz\n"
            f"Current Frequency: {cpu_freq.current:.2f}MHz"
        )
    
    def __str__(self) -> str:
        """Return a formatted string representation of the environment information."""
        return (
            f"{self.gpu_info}\n\n"
            f"CPU details: {self.cpu_info}\n\n"
            f"Detailed CPU information:\n{self.cpu_details}"
        )