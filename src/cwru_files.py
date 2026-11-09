"""List of the CWRU recordings used in this project.

Each recording is one .mat file from the CWRU Bearing Data Center.
Drive-end accelerometer, motor loads 0-3 HP, faults seeded by EDM.
"""

DOWNLOAD_URL = "https://engineering.case.edu/sites/default/files/{file_number}.mat"

CLASS_NAMES = ["Normal", "Inner Race", "Outer Race", "Ball"]

NORMAL_SAMPLING_RATE_HZ = 48000
FAULT_SAMPLING_RATE_HZ = 12000
TARGET_SAMPLING_RATE_HZ = 12000

# (file_number, class_name, fault_diameter_inches, motor_load_hp)
RECORDINGS = [
    (97, "Normal", 0.0, 0),
    (98, "Normal", 0.0, 1),
    (99, "Normal", 0.0, 2),
    (100, "Normal", 0.0, 3),

    (105, "Inner Race", 0.007, 0),
    (106, "Inner Race", 0.007, 1),
    (107, "Inner Race", 0.007, 2),
    (108, "Inner Race", 0.007, 3),
    (169, "Inner Race", 0.014, 0),
    (170, "Inner Race", 0.014, 1),
    (171, "Inner Race", 0.014, 2),
    (172, "Inner Race", 0.014, 3),
    (209, "Inner Race", 0.021, 0),
    (210, "Inner Race", 0.021, 1),
    (211, "Inner Race", 0.021, 2),
    (212, "Inner Race", 0.021, 3),

    (118, "Ball", 0.007, 0),
    (119, "Ball", 0.007, 1),
    (120, "Ball", 0.007, 2),
    (121, "Ball", 0.007, 3),
    (185, "Ball", 0.014, 0),
    (186, "Ball", 0.014, 1),
    (187, "Ball", 0.014, 2),
    (188, "Ball", 0.014, 3),
    (222, "Ball", 0.021, 0),
    (223, "Ball", 0.021, 1),
    (224, "Ball", 0.021, 2),
    (225, "Ball", 0.021, 3),

    # Outer race faults centered at the 6:00 position
    (130, "Outer Race", 0.007, 0),
    (131, "Outer Race", 0.007, 1),
    (132, "Outer Race", 0.007, 2),
    (133, "Outer Race", 0.007, 3),
    (197, "Outer Race", 0.014, 0),
    (198, "Outer Race", 0.014, 1),
    (199, "Outer Race", 0.014, 2),
    (200, "Outer Race", 0.014, 3),
    (234, "Outer Race", 0.021, 0),
    (235, "Outer Race", 0.021, 1),
    (236, "Outer Race", 0.021, 2),
    (237, "Outer Race", 0.021, 3),
]
