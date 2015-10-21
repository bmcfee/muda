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

    conda create -q -n $ENV_NAME "python=$1" $deps
}

if [ ! -f "$HOME/env/miniconda.sh" ]; then
    mkdir -p $HOME/env
    pushd $HOME/env
    
        # Download miniconda packages
        wget http://repo.continuum.io/miniconda/Miniconda-3.16.0-Linux-x86_64.sh -O miniconda.sh;

        # Install both environments
        src="$HOME/env/miniconda$TRAVIS_PYTHON_VERSION"
        bash miniconda.sh -b -p $src

        OLDPATH=$PATH
        export PATH="$src/bin:$PATH"
        conda_create $TRAVIS_PYTHON_VERSION

        source activate $ENV_NAME

        pip install python-coveralls
        pip install pysoundfile jsonpickle
        pip install --no-deps mir_eval audioread decorator librosa pyrubberband
        pip install --no-deps git+https://github.com/marl/jams.git

        source deactivate

        export PATH=$OLDPATH
    popd
else
    echo "Using cached dependencies"
fi
