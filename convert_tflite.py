- name: Train
        run: |
          python train_fanorona.py \
            --iterations ${{ inputs.iterations }} \
            --games-per-iteration ${{ inputs.games }} \
            --simulations ${{ inputs.simulations }} \
            --epochs ${{ inputs.epochs }} \
            --out checkpoints
            
      - name: Check checkpoint exists
        run: ls -la checkpoints/

      - name: Export TFLite
        run: python convert_tflite.py --checkpoint checkpoints/latest.keras --output fanorona_model.tflite