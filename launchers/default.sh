#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
#dt-exec python3 -m "my_package.my_script"
#dt-exec python3 -m "my_package.my_script2"
#dt-exec python3 -m "my_package.ps_code"
#dt-exec python3 -m "my_package.YETO"

python3 -m "my_package.cam_subscriber" &
python3 -m "my_package.pose_with_aruco" &
python3 -m "my_package.pose_without_aruco" &
dt-exec python3 -m "my_package.duckie_mover"

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
