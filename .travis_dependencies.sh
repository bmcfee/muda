#!/bin/sh

ENV_NAME="test-environment"
set -e

conda_create ()
{

    hash -r
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda config --add channels pypi
    conda info -a
    deps='pip numpy scipy nose pandas matplotlib scikit-learn'

    conda create -q -n $ENV_NAME "python=$TRAVIS_PYTHON_VERSION" $deps
}

src="$HOME/env/miniconda$TRAVIS_PYTHON_VERSION"
if [ ! -d "$src" ]; then
    mkdir -p $HOME/env
    pushd $HOME/env
    
        # Download miniconda packages
        wget http://repo.continuum.io/miniconda/Miniconda-3.16.0-Linux-x86_64.sh -O miniconda.sh;

        # Install both environments
        bash miniconda.sh -b -p $src

        export PATH="$src/bin:$PATH"
        conda_create

        source $src/bin/activate $ENV_NAME

        pip install python-coveralls
        pip install pysoundfile jsonpickle
        pip install mir_eval audioread decorator librosa pyrubberband
        pip install git+https://github.com/marl/jams.git

        source $src/bin/deactivate
    popd
else
    echo "Using cached dependencies"
fi
