from setuptools import find_packages, setup

package_name = "survivor_tagging_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # numpy and the survivor_tagging library (pip install -e <repo>) are runtime requirements, not ROS packages.
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="survivor tagging maintainer",
    maintainer_email="jainilagarwal20@gmail.com",
    description="ROS 2 wrapper of the survivor tagging library (adapters only in 4.1a)",
    license="TODO: licence not decided",
    tests_require=["pytest"],
    entry_points={"console_scripts": []},  # no nodes yet (substep 4.1b)
)
