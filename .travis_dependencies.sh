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
        bash miniconda.sh -b -p $HOME/env/miniconda27
        bash miniconda.sh -b -p $HOME/env/miniconda34
        bash miniconda.sh -b -p $HOME/env/miniconda35

        for version in 2.7 3.4 3.5 ; do
            if [[ "$version" == "2.7" ]]; then
                src="$HOME/env/miniconda27"
            elif [[ "$version" == "3.4" ]]; then
                src="$HOME/env/miniconda34"
            else
                src="$HOME/env/miniconda35"
            fi
            OLDPATH=$PATH
            export PATH="$src/bin:$PATH"
            conda_create $version

            source activate $ENV_NAME

            pip install python-coveralls
            pip install pysoundfile jsonpickle
            pip install git+https://github.com/marl/jams.git
            pip install audioread decorator
            pip install --no-deps librosa 
            pip install pyrubberband

            source deactivate

            export PATH=$OLDPATH
        done
    popd
else
    echo "Using cached dependencies"
fi
