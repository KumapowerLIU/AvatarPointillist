#!/usr/bin/env python3
"""
Batch script for combining NeRF datasets
Automatically processes expression sequences for all IDs in the specified directory,
generating merged datasets for each ID
"""

import os
import subprocess
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Batch combine NeRF datasets")
    parser.add_argument("--base_dir", type=str, 
                       default="/robby/share/Avatar/datasets/vhap_nersemble_v2_results/export",
                       help="Base directory path")
    parser.add_argument("--division_mode", type=str, 
                       choices=['random_single', 'random_group', 'last'],
                       default='random_group',
                       help="Data division mode")
    parser.add_argument("--dry_run", action="store_true",
                       help="Only show commands to be executed, do not actually execute")
    parser.add_argument("--vhap_script", type=str,
                       default="combine_nerf_datasets.py",
                       help="Path to combine_nerf_datasets.py script")
    parser.add_argument("--min_sequences", type=int, default=1,
                       help="Minimum number of sequences required to process an ID (default: 1)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    # Define all possible expression sequences
    sequences = ["EMO-1", "EMO-2", "EMO-3", "EMO-4", "EXP-2", "EXP-3", "EXP-4", "EXP-5", "EXP-8", "EXP-9", "FREE"]
    
    # Get all ID folders
    id_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    id_folders.sort()
    
    print(f"Found {len(id_folders)} ID folders: {[d.name for d in id_folders]}")
    
    successful_ids = []
    failed_ids = []
    
    for id_folder in id_folders:
        id_name = id_folder.name
        print(f"\nProcessing ID: {id_name}")
        
        # Check which sequences exist
        existing_sequences = []
        missing_sequences = []
        
        for seq in sequences:
            seq_path = id_folder / seq
            if seq_path.exists() and seq_path.is_dir():
                existing_sequences.append(seq_path)
            else:
                missing_sequences.append(seq)
        
        if missing_sequences:
            print(f"  Warning: Missing sequences {missing_sequences}")
        
        if len(existing_sequences) < args.min_sequences:
            print(f"  Skipping ID {id_name}: Available sequences ({len(existing_sequences)}) less than minimum required ({args.min_sequences})")
            failed_ids.append(id_name)
            continue
        elif len(existing_sequences) == 1:
            print(f"  ID {id_name} has only 1 sequence, will create single-sequence dataset")
        
        print(f"  Found {len(existing_sequences)} sequences: {[s.name for s in existing_sequences]}")
        
        # Generate target folder name
        seq_names = [s.name for s in existing_sequences]
        seq_names_str = "".join([s.replace("-", "") for s in seq_names])
        tgt_folder_name = f"UNION{len(existing_sequences)}_{id_name}_{seq_names_str}"
        tgt_folder = id_folder / tgt_folder_name
        
        # Build command
        cmd = [
            "python", args.vhap_script,
            "--src_folders"
        ]
        
        # Add all source folders
        for seq_path in existing_sequences:
            cmd.append(str(seq_path))
        
        cmd.extend([
            "--tgt_folder", str(tgt_folder),
            "--division_mode", args.division_mode
        ])
        
        print(f"  Target folder: {tgt_folder_name}")
        
        if args.dry_run:
            print(f"  [DRY RUN] Command to be executed:")
            print(f"    {' '.join(cmd)}")
        else:
            try:
                # Execute command
                print(f"  Executing merge...")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✓ Successfully merged ID {id_name}")
                successful_ids.append(id_name)
                
                # Optional: print the last few lines of output
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    for line in output_lines[-3:]:  # Show last 3 lines
                        print(f"    {line}")
                        
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to process ID {id_name}:")
                print(f"    Error code: {e.returncode}")
                if e.stderr:
                    print(f"    Error message: {e.stderr}")
                failed_ids.append(id_name)
            except Exception as e:
                print(f"  ✗ Exception occurred while processing ID {id_name}: {e}")
                failed_ids.append(id_name)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Successfully processed: {len(successful_ids)} IDs")
    if successful_ids:
        print(f"  {successful_ids}")
    
    print(f"Failed: {len(failed_ids)} IDs")
    if failed_ids:
        print(f"  {failed_ids}")
    
    if args.dry_run:
        print("\nNote: This is a dry run mode, no actual merge operations were executed")


if __name__ == "__main__":
    main()
