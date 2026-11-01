#map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>
module {
	func.func @sdsc_bundle(%arg_0_base_addr: !sdscbundle.input_arg<index>, %arg_1_base_addr: !sdscbundle.input_arg<index>, %arg_2_base_addr: !sdscbundle.input_arg<index>, %arg_3_base_addr: !sdscbundle.input_arg<index>, %arg_4_base_addr: !sdscbundle.input_arg<index>, %arg_5_base_addr: !sdscbundle.input_arg<index>, %arg_6_base_addr: !sdscbundle.input_arg<index>, %arg_7_base_addr: !sdscbundle.input_arg<index>, %arg_8_base_addr: !sdscbundle.input_arg<index>, %arg_9_base_addr: !sdscbundle.input_arg<index>, %arg_10_base_addr: !sdscbundle.input_arg<index>) {
		%pool = sdscbundle.device_mem_allocate 399360 bytes : index
		%arg_0 = sdscbundle.input_arg_extract value from %arg_0_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_1 = sdscbundle.input_arg_extract value from %arg_1_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_2 = sdscbundle.input_arg_extract value from %arg_2_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_3 = sdscbundle.input_arg_extract value from %arg_3_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_4 = sdscbundle.input_arg_extract value from %arg_4_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_5 = sdscbundle.input_arg_extract value from %arg_5_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_6 = sdscbundle.input_arg_extract value from %arg_6_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_7 = sdscbundle.input_arg_extract value from %arg_7_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_8 = sdscbundle.input_arg_extract value from %arg_8_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_9 = sdscbundle.input_arg_extract value from %arg_9_base_addr : !sdscbundle.input_arg<index> -> index
		%arg_10 = sdscbundle.input_arg_extract value from %arg_10_base_addr : !sdscbundle.input_arg<index> -> index
		%c0 = arith.constant 0 : index
		%c1 = arith.constant 1 : index
		%loop_bound_0 = arith.constant 2 : index
		%pool_offset_0 = arith.constant 0 : index
		%pool_addr_0 = arith.addi %pool, %pool_offset_0 : index
		%arg_4_core_offset_49280 = arith.constant 49280 : index
		%arg_4_core_49280 = arith.addi %arg_4, %arg_4_core_offset_49280 : index
		%arg_4_core_offset_98560 = arith.constant 98560 : index
		%arg_4_core_98560 = arith.addi %arg_4, %arg_4_core_offset_98560 : index
		%arg_4_core_offset_147840 = arith.constant 147840 : index
		%arg_4_core_147840 = arith.addi %arg_4, %arg_4_core_offset_147840 : index
		%arg_4_core_offset_197120 = arith.constant 197120 : index
		%arg_4_core_197120 = arith.addi %arg_4, %arg_4_core_offset_197120 : index
		%arg_4_core_offset_246400 = arith.constant 246400 : index
		%arg_4_core_246400 = arith.addi %arg_4, %arg_4_core_offset_246400 : index
		%arg_4_core_offset_295680 = arith.constant 295680 : index
		%arg_4_core_295680 = arith.addi %arg_4, %arg_4_core_offset_295680 : index
		%arg_4_core_offset_344960 = arith.constant 344960 : index
		%arg_4_core_344960 = arith.addi %arg_4, %arg_4_core_offset_344960 : index
		%pool_offset_4096 = arith.constant 4096 : index
		%pool_addr_4096 = arith.addi %pool, %pool_offset_4096 : index
		%pool_offset_4224 = arith.constant 4224 : index
		%pool_addr_4224 = arith.addi %pool, %pool_offset_4224 : index
		%pool_offset_4352 = arith.constant 4352 : index
		%pool_addr_4352 = arith.addi %pool, %pool_offset_4352 : index
		%pool_offset_4480 = arith.constant 4480 : index
		%pool_addr_4480 = arith.addi %pool, %pool_offset_4480 : index
		%pool_offset_4608 = arith.constant 4608 : index
		%pool_addr_4608 = arith.addi %pool, %pool_offset_4608 : index
		%pool_offset_4736 = arith.constant 4736 : index
		%pool_addr_4736 = arith.addi %pool, %pool_offset_4736 : index
		%pool_offset_4864 = arith.constant 4864 : index
		%pool_addr_4864 = arith.addi %pool, %pool_offset_4864 : index
		%pool_offset_4992 = arith.constant 4992 : index
		%pool_addr_4992 = arith.addi %pool, %pool_offset_4992 : index
		%pool_offset_398336 = arith.constant 398336 : index
		%pool_addr_398336 = arith.addi %pool, %pool_offset_398336 : index
		%pool_offset_398464 = arith.constant 398464 : index
		%pool_addr_398464 = arith.addi %pool, %pool_offset_398464 : index
		%pool_offset_398592 = arith.constant 398592 : index
		%pool_addr_398592 = arith.addi %pool, %pool_offset_398592 : index
		%pool_offset_398720 = arith.constant 398720 : index
		%pool_addr_398720 = arith.addi %pool, %pool_offset_398720 : index
		%pool_offset_398848 = arith.constant 398848 : index
		%pool_addr_398848 = arith.addi %pool, %pool_offset_398848 : index
		%pool_offset_398976 = arith.constant 398976 : index
		%pool_addr_398976 = arith.addi %pool, %pool_offset_398976 : index
		%pool_offset_399104 = arith.constant 399104 : index
		%pool_addr_399104 = arith.addi %pool, %pool_offset_399104 : index
		%pool_offset_399232 = arith.constant 399232 : index
		%pool_addr_399232 = arith.addi %pool, %pool_offset_399232 : index
		%arg_6_core_offset_6307840 = arith.constant 6307840 : index
		%arg_6_core_6307840 = arith.addi %arg_6, %arg_6_core_offset_6307840 : index
		%arg_6_core_offset_12615680 = arith.constant 12615680 : index
		%arg_6_core_12615680 = arith.addi %arg_6, %arg_6_core_offset_12615680 : index
		%arg_6_core_offset_18923520 = arith.constant 18923520 : index
		%arg_6_core_18923520 = arith.addi %arg_6, %arg_6_core_offset_18923520 : index
		%arg_6_core_offset_25231360 = arith.constant 25231360 : index
		%arg_6_core_25231360 = arith.addi %arg_6, %arg_6_core_offset_25231360 : index
		%arg_6_core_offset_31539200 = arith.constant 31539200 : index
		%arg_6_core_31539200 = arith.addi %arg_6, %arg_6_core_offset_31539200 : index
		%arg_6_core_offset_37847040 = arith.constant 37847040 : index
		%arg_6_core_37847040 = arith.addi %arg_6, %arg_6_core_offset_37847040 : index
		%arg_6_core_offset_44154880 = arith.constant 44154880 : index
		%arg_6_core_44154880 = arith.addi %arg_6, %arg_6_core_offset_44154880 : index
		%arg_7_core_offset_6307840 = arith.constant 6307840 : index
		%arg_7_core_6307840 = arith.addi %arg_7, %arg_7_core_offset_6307840 : index
		%arg_7_core_offset_12615680 = arith.constant 12615680 : index
		%arg_7_core_12615680 = arith.addi %arg_7, %arg_7_core_offset_12615680 : index
		%arg_7_core_offset_18923520 = arith.constant 18923520 : index
		%arg_7_core_18923520 = arith.addi %arg_7, %arg_7_core_offset_18923520 : index
		%arg_7_core_offset_25231360 = arith.constant 25231360 : index
		%arg_7_core_25231360 = arith.addi %arg_7, %arg_7_core_offset_25231360 : index
		%arg_7_core_offset_31539200 = arith.constant 31539200 : index
		%arg_7_core_31539200 = arith.addi %arg_7, %arg_7_core_offset_31539200 : index
		%arg_7_core_offset_37847040 = arith.constant 37847040 : index
		%arg_7_core_37847040 = arith.addi %arg_7, %arg_7_core_offset_37847040 : index
		%arg_7_core_offset_44154880 = arith.constant 44154880 : index
		%arg_7_core_44154880 = arith.addi %arg_7, %arg_7_core_offset_44154880 : index
		%pool_offset_512 = arith.constant 512 : index
		%pool_addr_512 = arith.addi %pool, %pool_offset_512 : index
		%pool_offset_1024 = arith.constant 1024 : index
		%pool_addr_1024 = arith.addi %pool, %pool_offset_1024 : index
		%pool_offset_1536 = arith.constant 1536 : index
		%pool_addr_1536 = arith.addi %pool, %pool_offset_1536 : index
		%pool_offset_2048 = arith.constant 2048 : index
		%pool_addr_2048 = arith.addi %pool, %pool_offset_2048 : index
		%pool_offset_2560 = arith.constant 2560 : index
		%pool_addr_2560 = arith.addi %pool, %pool_offset_2560 : index
		%pool_offset_3072 = arith.constant 3072 : index
		%pool_addr_3072 = arith.addi %pool, %pool_offset_3072 : index
		%pool_offset_3584 = arith.constant 3584 : index
		%pool_addr_3584 = arith.addi %pool, %pool_offset_3584 : index
		%pool_offset_128 = arith.constant 128 : index
		%pool_addr_128 = arith.addi %pool, %pool_offset_128 : index
		%pool_offset_256 = arith.constant 256 : index
		%pool_addr_256 = arith.addi %pool, %pool_offset_256 : index
		%pool_offset_384 = arith.constant 384 : index
		%pool_addr_384 = arith.addi %pool, %pool_offset_384 : index
		%pool_offset_640 = arith.constant 640 : index
		%pool_addr_640 = arith.addi %pool, %pool_offset_640 : index
		%pool_offset_768 = arith.constant 768 : index
		%pool_addr_768 = arith.addi %pool, %pool_offset_768 : index
		%pool_offset_896 = arith.constant 896 : index
		%pool_addr_896 = arith.addi %pool, %pool_offset_896 : index
		sdscbundle.sdsc_execute (%arg_0, %arg_1, %pool_addr_0) {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1, -2, -3]}
		sdscbundle.sdsc_execute (%arg_2) {sdsc_filename="sdsc_1.json", "symbol_ids"=[-4]}
		sdscbundle.sdsc_execute (%arg_3) {sdsc_filename="sdsc_2.json", "symbol_ids"=[-5]}
		sdscbundle.sdsc_execute (%arg_3) {sdsc_filename="sdsc_3.json", "symbol_ids"=[-6]}
		scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {
			sdscbundle.sdsc_execute (%arg_4, %arg_4_core_49280, %arg_4_core_98560, %arg_4_core_147840, %arg_4_core_197120, %arg_4_core_246400, %arg_4_core_295680, %arg_4_core_344960, %pool_addr_4096, %pool_addr_4224, %pool_addr_4352, %pool_addr_4480, %pool_addr_4608, %pool_addr_4736, %pool_addr_4864, %pool_addr_4992) {sdsc_filename="sdsc_4.json", "symbol_ids"=[-7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22]}
			%addr_0 = affine.apply #map_0(%i_0)[%arg_5]
			sdscbundle.sdsc_execute (%addr_0, %pool_addr_4096, %pool_addr_398336) {sdsc_filename="sdsc_5.json", "symbol_ids"=[-23, -24, -25]}
			sdscbundle.sdsc_execute (%pool_addr_398336, %pool_addr_398464, %pool_addr_398592, %pool_addr_398720, %pool_addr_398848, %pool_addr_398976, %pool_addr_399104, %pool_addr_399232, %arg_6, %arg_6_core_6307840, %arg_6_core_12615680, %arg_6_core_18923520, %arg_6_core_25231360, %arg_6_core_31539200, %arg_6_core_37847040, %arg_6_core_44154880) {sdsc_filename="sdsc_6.json", "symbol_ids"=[-26, -27, -28, -29, -30, -31, -32, -33, -34, -35, -36, -37, -38, -39, -40, -41]}
			sdscbundle.sdsc_execute (%pool_addr_398336, %pool_addr_398464, %pool_addr_398592, %pool_addr_398720, %pool_addr_398848, %pool_addr_398976, %pool_addr_399104, %pool_addr_399232, %arg_7, %arg_7_core_6307840, %arg_7_core_12615680, %arg_7_core_18923520, %arg_7_core_25231360, %arg_7_core_31539200, %arg_7_core_37847040, %arg_7_core_44154880) {sdsc_filename="sdsc_7.json", "symbol_ids"=[-42, -43, -44, -45, -46, -47, -48, -49, -50, -51, -52, -53, -54, -55, -56, -57]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_8.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute (%pool_addr_0, %pool_addr_512, %pool_addr_1024, %pool_addr_1536, %pool_addr_2048, %pool_addr_2560, %pool_addr_3072, %pool_addr_3584) {sdsc_filename="sdsc_9.json", "symbol_ids"=[-58, -59, -60, -61, -62, -63, -64, -65]}
			sdscbundle.sdsc_execute (%arg_8) {sdsc_filename="sdsc_10.json", "symbol_ids"=[-66]}
			%addr_1 = affine.apply #map_0(%i_0)[%arg_9]
			sdscbundle.sdsc_execute (%addr_1) {sdsc_filename="sdsc_11.json", "symbol_ids"=[-67]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_12.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_13.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_14.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_15.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_16.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_17.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_18.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_19.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_20.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_21.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_22.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_23.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_24.json", "symbol_ids"=[]}
			sdscbundle.sdsc_execute () {sdsc_filename="sdsc_25.json", "symbol_ids"=[]}
		}
		sdscbundle.sdsc_execute (%pool_addr_0, %pool_addr_128, %pool_addr_256, %pool_addr_384, %pool_addr_512, %pool_addr_640, %pool_addr_768, %pool_addr_896) {sdsc_filename="sdsc_26.json", "symbol_ids"=[-68, -69, -70, -71, -72, -73, -74, -75]}
		sdscbundle.sdsc_execute (%arg_0, %pool_addr_0, %arg_10) {sdsc_filename="sdsc_27.json", "symbol_ids"=[-76, -77, -78]}
		return
	}
}

