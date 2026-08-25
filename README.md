This project is a healthcare decision-support system designed to recommend suitable hospitals based on a patient's medical requirements.

The system uses a Mamdani Fuzzy Inference System (FIS) to evaluate and rank hospitals based on multiple factors such as distance, treatment cost, hospital rating/quality, and emergency capabilities. Fuzzy logic allows these factors to be handled as gradual values rather than strict yes/no conditions.

The hospital dataset is based on the Hospital Directory from the National Health Portal. The data has been cleaned and standardized, including hospital information, geographical coordinates, medical specialties, procedures, services, and facilities.

The system first identifies hospitals capable of handling the required medical specialty or procedure. Suitable hospitals are then evaluated using fuzzy membership functions and a set of fuzzy rules to produce an overall Hospital Suitability Score. Hospitals can subsequently be ranked based on this score.
