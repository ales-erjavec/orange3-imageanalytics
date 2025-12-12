#!/usr/bin/env python3
import os
from os import path, walk

from setuptools import setup
from setuptools.command.install import install


DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    # if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
    #     print("Directory '{}' does not exist. "
    #           "Please build documentation before running bdist_wheel."
    #           .format(path.abspath(local_dir)))
    #     sys.exit(0)
    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


class InstallMultilingualCommand(install):
    def run(self):
        install.run(self)
        self.compile_to_multilingual()

    def compile_to_multilingual(self):
        from trubar import translate

        package_dir = os.path.dirname(os.path.abspath(__file__))
        translate(
            "msgs.jaml",
            source_dir=os.path.join(self.install_lib, "orangecontrib", "imageanalytics"),
            config_file=os.path.join(package_dir, "i18n", "trubar-config.yaml"))


if __name__ == '__main__':
    include_documentation('doc/_build/html', 'help/orange3-imageanalytics')
    setup(
        cmdclass={
            'install': InstallMultilingualCommand,
        },
        data_files=DATA_FILES,
    )
