!/bin/bash

 # TPU Babysitter Script
 # This script monitors a TPU VM and recreates it if preempted, running a specific command only after recreation

 # Default configuration
 USERNAME="neel"
 PROJECT="asura-0"
 ZONE="us-central2-b"
 TYPE="v4-128"
 VM_IMAGE="tpu-vm-v4-base"
 PREEMPTIBLE=true
 AUTODELETE=true
 SUBNETWORK="default"
 USE_ALPHA=true
 RETRIES=-1  # -1 means infinite retries

 # Default TPU name
 DEFAULT_VM_NAME="smallnode"

 # Your specific command - with proper escaping
 DEFAULT_COMMAND='tmux kill-server; sudo rm -rf ./ReAct_Jax/ReAct/outputs/; sudo rm -rf *; curl -L https://gist.githubusercontent.com/neel04/9c26d460793466187b5dd8ffb2e4d90b/raw/run_uv.sh -o run.sh; sleep 1s && tmux new-session -d "source run.sh 2>&1 | tee out.log";'

 # Function to display usage
 function show_usage() {
   echo "Usage: tpu-babysitter.sh [VM_NAME] [options]"
   echo
   echo "Options:"
   echo "  -z, --zone ZONE             Zone to create the VM in (default: us-central2-b)"
   echo "  -t, --type TYPE             Type of TPU VM to create (default: v3-32)"
   echo "  -i, --image IMAGE           VM image to use (default: tpu-ubuntu2204-base)"
   echo "  -p, --preemptible           Use a preemptible VM (default: false)"
   echo "  -a, --no-autodelete         Don't delete the VM when done (default: autodelete=true)"
   echo "  -n, --subnetwork NETWORK    Subnetwork to use (default: default)"
   echo "  --use-alpha                 Use gcloud alpha (default: false)"
   echo "  --retries NUM               Number of retries before giving up. -1 for infinite (default: -1)"
   echo "  -c, --command \"COMMAND\"     Custom command to run after VM creation (will run in quotes)"
   echo
   echo "Example:"
   echo "  ./tpu-babysitter.sh"
   echo "  ./tpu-babysitter.sh custom-tpu-name -z us-central1-b -t v3-8 -p"
   exit 1
 }

 # Parse arguments
 VM_NAME=""
 COMMAND_TO_RUN="$DEFAULT_COMMAND"

 while [[ $# -gt 0 ]]; do
   key="$1"
   case $key in
     -z|--zone)
       ZONE="$2"
       shift 2
       ;;
     -t|--type)
       TYPE="$2"
       shift 2
       ;;
     -i|--image)
       VM_IMAGE="$2"
       shift 2
       ;;
     -p|--preemptible)
       PREEMPTIBLE="true"
       shift
       ;;
     -a|--no-autodelete)
       AUTODELETE="false"
       shift
       ;;
     -n|--subnetwork)
       SUBNETWORK="$2"
       shift 2
       ;;
     --use-alpha|--use_alpha)
       USE_ALPHA="true"
       shift
       ;;
     --retries)
       RETRIES="$2"
       shift 2
       ;;
     -c|--command)
       COMMAND_TO_RUN="$2"
       shift 2
       ;;
     -h|--help)
       show_usage
       ;;
     -*)
       echo "Error: Unknown option $1"
       show_usage
       ;;
     *)
       if [ -z "$VM_NAME" ]; then
         VM_NAME="$1"
         shift
       else
         echo "Error: VM name already set to $VM_NAME. Got $1 as well."
         show_usage
       fi
       ;;
   esac
 done

 # Set VM_NAME to default if not provided
 if [ -z "$VM_NAME" ]; then
   VM_NAME="$DEFAULT_VM_NAME"
   echo "Using default VM name: $VM_NAME"
 fi

 echo "TPU Babysitter starting for VM: $VM_NAME"
 echo "Will monitor VM in zone $ZONE"
 echo "Setup command: $COMMAND_TO_RUN"

 # Function to create a TPU VM
 function create_vm() {
   echo "Creating TPU VM $VM_NAME in zone $ZONE with type $TYPE..."

   # Construct the creation command
   local create_cmd="gcloud"
   if [ "$USE_ALPHA" == "true" ]; then
     create_cmd+=" alpha"
   fi

   create_cmd+=" compute tpus tpu-vm create $VM_NAME"
   create_cmd+=" --zone=$ZONE"
   create_cmd+=" --accelerator-type=$TYPE"
   create_cmd+=" --version=$VM_IMAGE"

   if [ "$PREEMPTIBLE" == "true" ]; then
     create_cmd+=" --preemptible"
   fi

   create_cmd+=" --subnetwork=$SUBNETWORK"

   # Execute the creation command
   echo "Running: $create_cmd"
   eval "$create_cmd"

   return $?
 }

 # Function to run command on VM
 function run_command_on_vm() {
   # Encode the command to base64 to handle special characters and quotes
   local encoded_cmd=$(echo "$COMMAND_TO_RUN" | base64 -w 0)

   echo "Running encoded command on VM $VM_NAME..."

   # Construct the SSH command
   local ssh_cmd="echo $encoded_cmd | base64 -d | bash"

   # First try running on all workers
   echo "\nAttempting to run command on all workers... username: $USERNAME@$VM_NAME"
   gcloud compute tpus tpu-vm ssh "$USERNAME@$VM_NAME" \
     --zone="$ZONE" \
     --worker=all \
     --command="$ssh_cmd"

   local exit_code=$?

   if [ $exit_code -ne 0 ]; then
     echo "Failed to run on all workers. !!! Manually investigate !!!"
   fi

   return $exit_code
 }

 # Function to create VM and run the command
 function create_and_setup_vm() {
   # First check if VM already exists and is ready
   gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" &> /dev/null
   if [ $? -eq 0 ]; then
     STATE=$(gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" | grep state | awk '{print $2}')
     if [ "$STATE" = "READY" ]; then
       echo "VM $VM_NAME already exists and is in READY state. Skipping creation."
       # Run the command on the existing VM
       echo "Running setup command on existing VM $VM_NAME..."
       run_command_on_vm
       return $?
     else
       echo "VM $VM_NAME exists but is in $STATE state. Will delete and recreate."
       yes | gcloud compute tpus tpu-vm delete --zone "$ZONE" "$VM_NAME" &> /dev/null
     fi
   fi

   create_vm

   if [ $? -ne 0 ]; then
     echo "Failed to create VM"
     return 1
   fi

   # Wait for VM to be fully ready
   echo "Waiting for VM to initialize..."

   # Check VM state repeatedly until it's READY
   local vm_ready=false
   local wait_count=0
   while [ "$vm_ready" = false ] && [ $wait_count -lt 12 ]; do
     sleep 30
     wait_count=$((wait_count + 1))

     gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" &> /dev/null
     if [ $? -eq 0 ]; then
       STATE=$(gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" | grep state | awk '{print $2}')
       if [ "$STATE" = "READY" ]; then
         vm_ready=true
         echo "VM is now in READY state"
         # Additional wait to ensure SSH is available
         echo "Waiting 30 more seconds for SSH to be available..."
         sleep 30
       else
         echo "VM state: $STATE (waiting... $wait_count/12)"
       fi
     else
       echo "VM not found yet (waiting... $wait_count/12)"
     fi
   done

   if [ "$vm_ready" = false ]; then
     echo "VM did not reach READY state in a reasonable time"
     return 1
   fi

   # Run the command on the newly created VM
   echo "Running setup command on VM $VM_NAME..."
   run_command_on_vm
   local cmd_exit_code=$?

   if [ $cmd_exit_code -eq 0 ]; then
     echo "Setup command executed successfully"
   else
     echo "Setup command failed with exit code $cmd_exit_code"
   fi

   return $cmd_exit_code
 }

 # Function to check VM status
 function check_vm_status() {
   gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" &> /dev/null
   if [ $? -ne 0 ]; then
     echo "VM $VM_NAME does not exist"
     return 1
   fi

   STATE=$(gcloud compute tpus tpu-vm describe --zone "$ZONE" "$VM_NAME" | grep state | awk '{print $2}')
   echo "VM $VM_NAME state: $STATE"

   if [ "$STATE" != "READY" ]; then
     return 1
   fi

   return 0
 }

 # Main monitoring loop
 NEEDS_SETUP=true
 RETRY_COUNT=0

 while true; do
   # First check if VM needs to be created initially
   if $NEEDS_SETUP; then
     echo "Initial setup needed for VM $VM_NAME"
     create_and_setup_vm
     COMMAND_EXIT_CODE=$?

     if [ $COMMAND_EXIT_CODE -eq 0 ]; then
       echo "Initial setup completed successfully"
       NEEDS_SETUP=false
     else
       echo "Initial setup failed with exit code $COMMAND_EXIT_CODE"
       RETRY_COUNT=$((RETRY_COUNT + 1))

       if [ "$RETRIES" -ge 0 ] && [ $RETRY_COUNT -ge "$RETRIES" ]; then
         echo "Failed $RETRY_COUNT times, giving up."
         exit 1
       fi

       echo "Retrying in 30 seconds... (attempt $RETRY_COUNT)"
       sleep 30
       continue
     fi
   fi

   # Now monitor for preemption
   check_vm_status
   if [ $? -ne 0 ]; then
     echo "VM $VM_NAME is not in READY state or doesn't exist"
     echo "Deleting VM if it exists..."
     yes | gcloud compute tpus tpu-vm delete --zone "$ZONE" "$VM_NAME" &> /dev/null

     echo "VM was preempted or crashed. Recreating and running setup command..."
     create_and_setup_vm
     COMMAND_EXIT_CODE=$?

     if [ $COMMAND_EXIT_CODE -ne 0 ]; then
       echo "Setup after preemption failed with exit code $COMMAND_EXIT_CODE"
       RETRY_COUNT=$((RETRY_COUNT + 1))

       if [ "$RETRIES" -ge 0 ] && [ $RETRY_COUNT -ge "$RETRIES" ]; then
         echo "Failed $RETRY_COUNT times, giving up."
         exit 1
       fi
     else
       echo "VM recreated and setup command executed successfully"
       RETRY_COUNT=0  # Reset retry count after successful recovery
     fi
   else
     echo "VM $VM_NAME is running normally"
   fi

   echo "Next check in 60 seconds..."
   sleep 60
 done
