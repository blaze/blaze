define void @streamufunc___numba_specialized_add_int32_int32(i8** noalias nocapture %args, i64* noalias nocapture %dimensions, i64* noalias nocapture %steps, i8* noalias nocapture %data) nounwind {
decl:
  %0 = load i64* %dimensions
  %1 = load i64* %steps
  %2 = getelementptr i64* %steps, i32 1
  %3 = load i64* %2
  %4 = getelementptr i64* %steps, i32 2
  %5 = load i64* %4
  %6 = icmp sgt i64 %0, 0
  br i1 %6, label %loop.body.lr.ph, label %loop.end

loop.body.lr.ph:                                  ; preds = %decl
  %7 = getelementptr i8** %args, i32 2
  %8 = getelementptr i8** %args, i32 1
  %9 = load i8** %7
  %10 = load i8** %8
  %11 = load i8** %args
  %.sum.v.i0.1 = insertelement <2 x i64> undef, i64 %1, i32 0
  %.sum.v.i0.2 = insertelement <2 x i64> %.sum.v.i0.1, i64 %3, i32 1
  %.sum = shl <2 x i64> %.sum.v.i0.2, <i64 1, i64 1>
  %.sum.v.r1 = extractelement <2 x i64> %.sum, i32 0
  %.sum.v.r2 = extractelement <2 x i64> %.sum, i32 1
  %.sum52 = add <2 x i64> %.sum, %.sum.v.i0.2
  %.sum52.v.r1 = extractelement <2 x i64> %.sum52, i32 0
  %.sum52.v.r2 = extractelement <2 x i64> %.sum52, i32 1
  %.sum53 = add <2 x i64> %.sum52, %.sum.v.i0.2
  %.sum53.v.r1 = extractelement <2 x i64> %.sum53, i32 0
  %.sum53.v.r2 = extractelement <2 x i64> %.sum53, i32 1
  %.sum54 = add <2 x i64> %.sum53, %.sum.v.i0.2
  %.sum54.v.r1 = extractelement <2 x i64> %.sum54, i32 0
  %.sum54.v.r2 = extractelement <2 x i64> %.sum54, i32 1
  %.sum55 = add <2 x i64> %.sum54, %.sum.v.i0.2
  %.sum55.v.r1 = extractelement <2 x i64> %.sum55, i32 0
  %.sum55.v.r2 = extractelement <2 x i64> %.sum55, i32 1
  %.sum56 = add <2 x i64> %.sum55, %.sum.v.i0.2
  %.sum56.v.r1 = extractelement <2 x i64> %.sum56, i32 0
  %.sum56.v.r2 = extractelement <2 x i64> %.sum56, i32 1
  %.sum57 = add <2 x i64> %.sum56, %.sum.v.i0.2
  %.sum57.v.r1 = extractelement <2 x i64> %.sum57, i32 0
  %.sum57.v.r2 = extractelement <2 x i64> %.sum57, i32 1
  %.sum65 = shl i64 %5, 1
  %.sum66 = add i64 %.sum65, %5
  %.sum67 = add i64 %.sum66, %5
  %.sum68 = add i64 %.sum67, %5
  %.sum69 = add i64 %.sum68, %5
  %.sum70 = add i64 %.sum69, %5
  %.sum71 = add i64 %.sum70, %5
  br label %loop.body

loop.body:                                        ; preds = %loop.body.lr.ph, %if.end31
  %lsr.iv = phi i64 [ %0, %loop.body.lr.ph ], [ %lsr.iv.next, %if.end31 ]
  %.03278 = phi i64 [ 0, %loop.body.lr.ph ], [ %166, %if.end31 ]
  %.03377 = phi i8* [ %9, %loop.body.lr.ph ], [ %.8, %if.end31 ]
  %.03476 = phi i8* [ %10, %loop.body.lr.ph ], [ %.842, %if.end31 ]
  %.04375 = phi i8* [ %11, %loop.body.lr.ph ], [ %.851, %if.end31 ]
  %12 = icmp slt i64 %lsr.iv, 8
  %. = select i1 %12, i64 %lsr.iv, i64 8
  %13 = icmp eq i64 %., 8
  br i1 %13, label %if.then5, label %if.else6

loop.end:                                         ; preds = %if.end31, %decl
  ret void

if.then5:                                         ; preds = %loop.body
  %14 = bitcast i8* %.04375 to i32*
  %15 = load i32* %14
  %16 = getelementptr i8* %.04375, i64 %1
  %17 = bitcast i8* %16 to i32*
  %18 = load i32* %17
  %19 = getelementptr i8* %.04375, i64 %.sum.v.r1
  %20 = bitcast i8* %19 to i32*
  %21 = load i32* %20
  %22 = getelementptr i8* %.04375, i64 %.sum52.v.r1
  %23 = bitcast i8* %22 to i32*
  %24 = load i32* %23
  %25 = getelementptr i8* %.04375, i64 %.sum53.v.r1
  %26 = bitcast i8* %25 to i32*
  %27 = load i32* %26
  %28 = getelementptr i8* %.04375, i64 %.sum54.v.r1
  %29 = bitcast i8* %28 to i32*
  %30 = load i32* %29
  %31 = getelementptr i8* %.04375, i64 %.sum55.v.r1
  %32 = bitcast i8* %31 to i32*
  %33 = load i32* %32
  %34 = getelementptr i8* %.04375, i64 %.sum56.v.r1
  %35 = bitcast i8* %34 to i32*
  %36 = load i32* %35
  %37 = getelementptr i8* %.04375, i64 %.sum57.v.r1
  %38 = bitcast i8* %.03476 to i32*
  %39 = load i32* %38
  %40 = getelementptr i8* %.03476, i64 %3
  %41 = bitcast i8* %40 to i32*
  %42 = load i32* %41
  %43 = getelementptr i8* %.03476, i64 %.sum.v.r2
  %44 = bitcast i8* %43 to i32*
  %45 = load i32* %44
  %46 = getelementptr i8* %.03476, i64 %.sum52.v.r2
  %47 = bitcast i8* %46 to i32*
  %48 = load i32* %47
  %49 = getelementptr i8* %.03476, i64 %.sum53.v.r2
  %50 = bitcast i8* %49 to i32*
  %51 = load i32* %50
  %52 = getelementptr i8* %.03476, i64 %.sum54.v.r2
  %53 = bitcast i8* %52 to i32*
  %54 = load i32* %53
  %55 = getelementptr i8* %.03476, i64 %.sum55.v.r2
  %56 = bitcast i8* %55 to i32*
  %57 = load i32* %56
  %58 = getelementptr i8* %.03476, i64 %.sum56.v.r2
  %59 = bitcast i8* %58 to i32*
  %60 = load i32* %59
  %61 = getelementptr i8* %.03476, i64 %.sum57.v.r2
  %62 = add i32 %39, %15
  %63 = add i32 %42, %18
  %64 = add i32 %45, %21
  %65 = add i32 %48, %24
  %66 = add i32 %51, %27
  %67 = add i32 %54, %30
  %68 = add i32 %57, %33
  %69 = add i32 %60, %36
  %70 = bitcast i8* %.03377 to i32*
  store i32 %62, i32* %70
  %71 = getelementptr i8* %.03377, i64 %5
  %72 = bitcast i8* %71 to i32*
  store i32 %63, i32* %72
  %73 = getelementptr i8* %.03377, i64 %.sum65
  %74 = bitcast i8* %73 to i32*
  store i32 %64, i32* %74
  %75 = getelementptr i8* %.03377, i64 %.sum66
  %76 = bitcast i8* %75 to i32*
  store i32 %65, i32* %76
  %77 = getelementptr i8* %.03377, i64 %.sum67
  %78 = bitcast i8* %77 to i32*
  store i32 %66, i32* %78
  %79 = getelementptr i8* %.03377, i64 %.sum68
  %80 = bitcast i8* %79 to i32*
  store i32 %67, i32* %80
  %81 = getelementptr i8* %.03377, i64 %.sum69
  %82 = bitcast i8* %81 to i32*
  store i32 %68, i32* %82
  %83 = getelementptr i8* %.03377, i64 %.sum70
  %84 = bitcast i8* %83 to i32*
  store i32 %69, i32* %84
  %85 = getelementptr i8* %.03377, i64 %.sum71
  br label %if.end31

if.else6:                                         ; preds = %loop.body
  %86 = icmp sgt i64 %., 0
  br i1 %86, label %if.end9, label %if.end31

if.end9:                                          ; preds = %if.else6
  %87 = bitcast i8* %.04375 to i32*
  %88 = load i32* %87
  %89 = getelementptr i8* %.04375, i64 %1
  %90 = bitcast i8* %.03476 to i32*
  %91 = load i32* %90
  %92 = getelementptr i8* %.03476, i64 %3
  %93 = add i32 %91, %88
  %94 = bitcast i8* %.03377 to i32*
  store i32 %93, i32* %94, !nontemporal !0
  %95 = getelementptr i8* %.03377, i64 %5
  %96 = icmp sgt i64 %., 1
  br i1 %96, label %if.then10, label %if.end12

if.then10:                                        ; preds = %if.end9
  %97 = bitcast i8* %89 to i32*
  %98 = load i32* %97
  %99 = getelementptr i8* %.04375, i64 %.sum.v.r1
  %100 = bitcast i8* %92 to i32*
  %101 = load i32* %100
  %102 = getelementptr i8* %.03476, i64 %.sum.v.r2
  %103 = add i32 %101, %98
  %104 = bitcast i8* %95 to i32*
  store i32 %103, i32* %104, !nontemporal !0
  %105 = getelementptr i8* %.03377, i64 %.sum65
  br label %if.end12

if.end12:                                         ; preds = %if.end9, %if.then10
  %.245 = phi i8* [ %99, %if.then10 ], [ %89, %if.end9 ]
  %.236 = phi i8* [ %102, %if.then10 ], [ %92, %if.end9 ]
  %.2 = phi i8* [ %105, %if.then10 ], [ %95, %if.end9 ]
  %106 = icmp sgt i64 %., 2
  br i1 %106, label %if.then13, label %if.end15

if.then13:                                        ; preds = %if.end12
  %107 = bitcast i8* %.245 to i32*
  %108 = load i32* %107
  %109 = getelementptr i8* %.245, i64 %1
  %110 = bitcast i8* %.236 to i32*
  %111 = load i32* %110
  %112 = getelementptr i8* %.236, i64 %3
  %113 = add i32 %111, %108
  %114 = bitcast i8* %.2 to i32*
  store i32 %113, i32* %114, !nontemporal !0
  %115 = getelementptr i8* %.2, i64 %5
  br label %if.end15

if.end15:                                         ; preds = %if.end12, %if.then13
  %.346 = phi i8* [ %109, %if.then13 ], [ %.245, %if.end12 ]
  %.337 = phi i8* [ %112, %if.then13 ], [ %.236, %if.end12 ]
  %.3 = phi i8* [ %115, %if.then13 ], [ %.2, %if.end12 ]
  %116 = icmp sgt i64 %., 3
  br i1 %116, label %if.then16, label %if.end18

if.then16:                                        ; preds = %if.end15
  %117 = bitcast i8* %.346 to i32*
  %118 = load i32* %117
  %119 = getelementptr i8* %.346, i64 %1
  %120 = bitcast i8* %.337 to i32*
  %121 = load i32* %120
  %122 = getelementptr i8* %.337, i64 %3
  %123 = add i32 %121, %118
  %124 = bitcast i8* %.3 to i32*
  store i32 %123, i32* %124, !nontemporal !0
  %125 = getelementptr i8* %.3, i64 %5
  br label %if.end18

if.end18:                                         ; preds = %if.end15, %if.then16
  %.447 = phi i8* [ %119, %if.then16 ], [ %.346, %if.end15 ]
  %.438 = phi i8* [ %122, %if.then16 ], [ %.337, %if.end15 ]
  %.4 = phi i8* [ %125, %if.then16 ], [ %.3, %if.end15 ]
  %126 = icmp sgt i64 %., 4
  br i1 %126, label %if.then19, label %if.end21

if.then19:                                        ; preds = %if.end18
  %127 = bitcast i8* %.447 to i32*
  %128 = load i32* %127
  %129 = getelementptr i8* %.447, i64 %1
  %130 = bitcast i8* %.438 to i32*
  %131 = load i32* %130
  %132 = getelementptr i8* %.438, i64 %3
  %133 = add i32 %131, %128
  %134 = bitcast i8* %.4 to i32*
  store i32 %133, i32* %134, !nontemporal !0
  %135 = getelementptr i8* %.4, i64 %5
  br label %if.end21

if.end21:                                         ; preds = %if.end18, %if.then19
  %.548 = phi i8* [ %129, %if.then19 ], [ %.447, %if.end18 ]
  %.539 = phi i8* [ %132, %if.then19 ], [ %.438, %if.end18 ]
  %.5 = phi i8* [ %135, %if.then19 ], [ %.4, %if.end18 ]
  %136 = icmp sgt i64 %., 5
  br i1 %136, label %if.then22, label %if.end24

if.then22:                                        ; preds = %if.end21
  %137 = bitcast i8* %.548 to i32*
  %138 = load i32* %137
  %139 = getelementptr i8* %.548, i64 %1
  %140 = bitcast i8* %.539 to i32*
  %141 = load i32* %140
  %142 = getelementptr i8* %.539, i64 %3
  %143 = add i32 %141, %138
  %144 = bitcast i8* %.5 to i32*
  store i32 %143, i32* %144, !nontemporal !0
  %145 = getelementptr i8* %.5, i64 %5
  br label %if.end24

if.end24:                                         ; preds = %if.end21, %if.then22
  %.649 = phi i8* [ %139, %if.then22 ], [ %.548, %if.end21 ]
  %.640 = phi i8* [ %142, %if.then22 ], [ %.539, %if.end21 ]
  %.6 = phi i8* [ %145, %if.then22 ], [ %.5, %if.end21 ]
  %146 = icmp sgt i64 %., 6
  br i1 %146, label %if.then25, label %if.end27

if.then25:                                        ; preds = %if.end24
  %147 = bitcast i8* %.649 to i32*
  %148 = load i32* %147
  %149 = getelementptr i8* %.649, i64 %1
  %150 = bitcast i8* %.640 to i32*
  %151 = load i32* %150
  %152 = getelementptr i8* %.640, i64 %3
  %153 = add i32 %151, %148
  %154 = bitcast i8* %.6 to i32*
  store i32 %153, i32* %154, !nontemporal !0
  %155 = getelementptr i8* %.6, i64 %5
  br label %if.end27

if.end27:                                         ; preds = %if.end24, %if.then25
  %.750 = phi i8* [ %149, %if.then25 ], [ %.649, %if.end24 ]
  %.741 = phi i8* [ %152, %if.then25 ], [ %.640, %if.end24 ]
  %.7 = phi i8* [ %155, %if.then25 ], [ %.6, %if.end24 ]
  %156 = icmp sgt i64 %., 7
  br i1 %156, label %if.then28, label %if.end31

if.then28:                                        ; preds = %if.end27
  %157 = bitcast i8* %.750 to i32*
  %158 = load i32* %157
  %159 = getelementptr i8* %.750, i64 %1
  %160 = bitcast i8* %.741 to i32*
  %161 = load i32* %160
  %162 = getelementptr i8* %.741, i64 %3
  %163 = add i32 %161, %158
  %164 = bitcast i8* %.7 to i32*
  store i32 %163, i32* %164, !nontemporal !0
  %165 = getelementptr i8* %.7, i64 %5
  br label %if.end31

if.end31:                                         ; preds = %if.else6, %if.then28, %if.end27, %if.then5
  %.851 = phi i8* [ %37, %if.then5 ], [ %159, %if.then28 ], [ %.750, %if.end27 ], [ %.04375, %if.else6 ]
  %.842 = phi i8* [ %61, %if.then5 ], [ %162, %if.then28 ], [ %.741, %if.end27 ], [ %.03476, %if.else6 ]
  %.8 = phi i8* [ %85, %if.then5 ], [ %165, %if.then28 ], [ %.7, %if.end27 ], [ %.03377, %if.else6 ]
  %166 = add i64 %.03278, 8
  %lsr.iv.next = add i64 %lsr.iv, -8
  %167 = icmp slt i64 %166, %0
  br i1 %167, label %loop.body, label %loop.end
}
