Environment Configuration
=========================================


Download code from github
-------------------------------

.. code-block:: bash

    git clone tpu-mq.git_project

After the code has been downloaded, we recommend compiling it using our Docker environment. 
For Docker configuration, please refer to the subsequent sections.



Basic Environment Configuration
---------------------------------
Download the required image from DockerHub :


.. code-block:: shell

   docker pull tpuc_dev:v3.4.6-cuda


Make sure the installation package is in the current directory, and then create a container in the current directory as follows:


.. code-block:: shell

    docker run --privileged --name myname -v $PWD:/workspace -it tpuc_dev:v3.4.6-cuda
    # "myname" is just an example, you can use any name you want


In the running Docker container, compile tpu-mq using the following command:

.. code-block:: shell

    cd tpu-mq
    python setup.py install
