define void @streamufunc___numba_specialized_add1_int32(i8** noalias nocapture %args, i64* noalias nocapture %dimensions, i64* noalias nocapture %steps, i8* noalias nocapture %data) nounwind {
decl:
  %0 = load i64* %dimensions
  %1 = load i64* %steps
  %2 = getelementptr i64* %steps, i32 1
  %3 = load i64* %2
  %4 = icmp sgt i64 %0, 0
  br i1 %4, label %loop.body.lr.ph, label %loop.end

loop.body.lr.ph:                                  ; preds = %decl
  %5 = getelementptr i8** %args, i32 1
  %6 = load i8** %5
  %7 = load i8** %args
  %.sum.v.i0.1 = insertelement <2 x i64> undef, i64 %1, i32 0
  %.sum.v.i0.2 = insertelement <2 x i64> %.sum.v.i0.1, i64 %3, i32 1
  %.sum = shl <2 x i64> %.sum.v.i0.2, <i64 1, i64 1>
  %.sum.v.r1 = extractelement <2 x i64> %.sum, i32 0
  %.sum.v.r2 = extractelement <2 x i64> %.sum, i32 1
  %.sum43 = add <2 x i64> %.sum, %.sum.v.i0.2
  %.sum43.v.r1 = extractelement <2 x i64> %.sum43, i32 0
  %.sum43.v.r2 = extractelement <2 x i64> %.sum43, i32 1
  %.sum44 = add <2 x i64> %.sum43, %.sum.v.i0.2
  %.sum44.v.r1 = extractelement <2 x i64> %.sum44, i32 0
  %.sum44.v.r2 = extractelement <2 x i64> %.sum44, i32 1
  %.sum45 = add <2 x i64> %.sum44, %.sum.v.i0.2
  %.sum45.v.r1 = extractelement <2 x i64> %.sum45, i32 0
  %.sum45.v.r2 = extractelement <2 x i64> %.sum45, i32 1
  %.sum46 = add <2 x i64> %.sum45, %.sum.v.i0.2
  %.sum46.v.r1 = extractelement <2 x i64> %.sum46, i32 0
  %.sum46.v.r2 = extractelement <2 x i64> %.sum46, i32 1
  %.sum47 = add <2 x i64> %.sum46, %.sum.v.i0.2
  %.sum47.v.r1 = extractelement <2 x i64> %.sum47, i32 0
  %.sum47.v.r2 = extractelement <2 x i64> %.sum47, i32 1
  %.sum48 = add <2 x i64> %.sum47, %.sum.v.i0.2
  %.sum48.v.r1 = extractelement <2 x i64> %.sum48, i32 0
  %.sum48.v.r2 = extractelement <2 x i64> %.sum48, i32 1
  br label %loop.body

loop.body:                                        ; preds = %loop.body.lr.ph, %if.end31
  %lsr.iv = phi i64 [ %0, %loop.body.lr.ph ], [ %lsr.iv.next, %if.end31 ]
  %.03260 = phi i64 [ 0, %loop.body.lr.ph ], [ %114, %if.end31 ]
  %.03359 = phi i8* [ %6, %loop.body.lr.ph ], [ %.8, %if.end31 ]
  %.03458 = phi i8* [ %7, %loop.body.lr.ph ], [ %.842, %if.end31 ]
  %8 = icmp slt i64 %lsr.iv, 8
  %. = select i1 %8, i64 %lsr.iv, i64 8
  %9 = icmp eq i64 %., 8
  br i1 %9, label %if.then5, label %if.else6

loop.end:                                         ; preds = %if.end31, %decl
  ret void

if.then5:                                         ; preds = %loop.body
  %10 = bitcast i8* %.03458 to i32*
  %11 = load i32* %10
  %12 = getelementptr i8* %.03458, i64 %1
  %13 = bitcast i8* %12 to i32*
  %14 = load i32* %13
  %15 = getelementptr i8* %.03458, i64 %.sum.v.r1
  %16 = bitcast i8* %15 to i32*
  %17 = load i32* %16
  %18 = getelementptr i8* %.03458, i64 %.sum43.v.r1
  %19 = bitcast i8* %18 to i32*
  %20 = load i32* %19
  %21 = getelementptr i8* %.03458, i64 %.sum44.v.r1
  %22 = bitcast i8* %21 to i32*
  %23 = load i32* %22
  %24 = getelementptr i8* %.03458, i64 %.sum45.v.r1
  %25 = bitcast i8* %24 to i32*
  %26 = load i32* %25
  %27 = getelementptr i8* %.03458, i64 %.sum46.v.r1
  %28 = bitcast i8* %27 to i32*
  %29 = load i32* %28
  %30 = getelementptr i8* %.03458, i64 %.sum47.v.r1
  %31 = bitcast i8* %30 to i32*
  %32 = load i32* %31
  %33 = getelementptr i8* %.03458, i64 %.sum48.v.r1
  %34 = add i32 %11, 2
  %35 = add i32 %14, 2
  %36 = add i32 %17, 2
  %37 = add i32 %20, 2
  %38 = add i32 %23, 2
  %39 = add i32 %26, 2
  %40 = add i32 %29, 2
  %41 = add i32 %32, 2
  %42 = bitcast i8* %.03359 to i32*
  store i32 %34, i32* %42
  %43 = getelementptr i8* %.03359, i64 %3
  %44 = bitcast i8* %43 to i32*
  store i32 %35, i32* %44
  %45 = getelementptr i8* %.03359, i64 %.sum.v.r2
  %46 = bitcast i8* %45 to i32*
  store i32 %36, i32* %46
  %47 = getelementptr i8* %.03359, i64 %.sum43.v.r2
  %48 = bitcast i8* %47 to i32*
  store i32 %37, i32* %48
  %49 = getelementptr i8* %.03359, i64 %.sum44.v.r2
  %50 = bitcast i8* %49 to i32*
  store i32 %38, i32* %50
  %51 = getelementptr i8* %.03359, i64 %.sum45.v.r2
  %52 = bitcast i8* %51 to i32*
  store i32 %39, i32* %52
  %53 = getelementptr i8* %.03359, i64 %.sum46.v.r2
  %54 = bitcast i8* %53 to i32*
  store i32 %40, i32* %54
  %55 = getelementptr i8* %.03359, i64 %.sum47.v.r2
  %56 = bitcast i8* %55 to i32*
  store i32 %41, i32* %56
  %57 = getelementptr i8* %.03359, i64 %.sum48.v.r2
  br label %if.end31

if.else6:                                         ; preds = %loop.body
  %58 = icmp sgt i64 %., 0
  br i1 %58, label %if.end9, label %if.end31

if.end9:                                          ; preds = %if.else6
  %59 = bitcast i8* %.03458 to i32*
  %60 = load i32* %59
  %61 = getelementptr i8* %.03458, i64 %1
  %62 = add i32 %60, 2
  %63 = bitcast i8* %.03359 to i32*
  store i32 %62, i32* %63, !nontemporal !0
  %64 = getelementptr i8* %.03359, i64 %3
  %65 = icmp sgt i64 %., 1
  br i1 %65, label %if.then10, label %if.end12

if.then10:                                        ; preds = %if.end9
  %66 = bitcast i8* %61 to i32*
  %67 = load i32* %66
  %68 = getelementptr i8* %.03458, i64 %.sum.v.r1
  %69 = add i32 %67, 2
  %70 = bitcast i8* %64 to i32*
  store i32 %69, i32* %70, !nontemporal !0
  %71 = getelementptr i8* %.03359, i64 %.sum.v.r2
  br label %if.end12

if.end12:                                         ; preds = %if.end9, %if.then10
  %.236 = phi i8* [ %68, %if.then10 ], [ %61, %if.end9 ]
  %.2 = phi i8* [ %71, %if.then10 ], [ %64, %if.end9 ]
  %72 = icmp sgt i64 %., 2
  br i1 %72, label %if.then13, label %if.end15

if.then13:                                        ; preds = %if.end12
  %73 = bitcast i8* %.236 to i32*
  %74 = load i32* %73
  %75 = getelementptr i8* %.236, i64 %1
  %76 = add i32 %74, 2
  %77 = bitcast i8* %.2 to i32*
  store i32 %76, i32* %77, !nontemporal !0
  %78 = getelementptr i8* %.2, i64 %3
  br label %if.end15

if.end15:                                         ; preds = %if.end12, %if.then13
  %.337 = phi i8* [ %75, %if.then13 ], [ %.236, %if.end12 ]
  %.3 = phi i8* [ %78, %if.then13 ], [ %.2, %if.end12 ]
  %79 = icmp sgt i64 %., 3
  br i1 %79, label %if.then16, label %if.end18

if.then16:                                        ; preds = %if.end15
  %80 = bitcast i8* %.337 to i32*
  %81 = load i32* %80
  %82 = getelementptr i8* %.337, i64 %1
  %83 = add i32 %81, 2
  %84 = bitcast i8* %.3 to i32*
  store i32 %83, i32* %84, !nontemporal !0
  %85 = getelementptr i8* %.3, i64 %3
  br label %if.end18

if.end18:                                         ; preds = %if.end15, %if.then16
  %.438 = phi i8* [ %82, %if.then16 ], [ %.337, %if.end15 ]
  %.4 = phi i8* [ %85, %if.then16 ], [ %.3, %if.end15 ]
  %86 = icmp sgt i64 %., 4
  br i1 %86, label %if.then19, label %if.end21

if.then19:                                        ; preds = %if.end18
  %87 = bitcast i8* %.438 to i32*
  %88 = load i32* %87
  %89 = getelementptr i8* %.438, i64 %1
  %90 = add i32 %88, 2
  %91 = bitcast i8* %.4 to i32*
  store i32 %90, i32* %91, !nontemporal !0
  %92 = getelementptr i8* %.4, i64 %3
  br label %if.end21

if.end21:                                         ; preds = %if.end18, %if.then19
  %.539 = phi i8* [ %89, %if.then19 ], [ %.438, %if.end18 ]
  %.5 = phi i8* [ %92, %if.then19 ], [ %.4, %if.end18 ]
  %93 = icmp sgt i64 %., 5
  br i1 %93, label %if.then22, label %if.end24

if.then22:                                        ; preds = %if.end21
  %94 = bitcast i8* %.539 to i32*
  %95 = load i32* %94
  %96 = getelementptr i8* %.539, i64 %1
  %97 = add i32 %95, 2
  %98 = bitcast i8* %.5 to i32*
  store i32 %97, i32* %98, !nontemporal !0
  %99 = getelementptr i8* %.5, i64 %3
  br label %if.end24

if.end24:                                         ; preds = %if.end21, %if.then22
  %.640 = phi i8* [ %96, %if.then22 ], [ %.539, %if.end21 ]
  %.6 = phi i8* [ %99, %if.then22 ], [ %.5, %if.end21 ]
  %100 = icmp sgt i64 %., 6
  br i1 %100, label %if.then25, label %if.end27

if.then25:                                        ; preds = %if.end24
  %101 = bitcast i8* %.640 to i32*
  %102 = load i32* %101
  %103 = getelementptr i8* %.640, i64 %1
  %104 = add i32 %102, 2
  %105 = bitcast i8* %.6 to i32*
  store i32 %104, i32* %105, !nontemporal !0
  %106 = getelementptr i8* %.6, i64 %3
  br label %if.end27

if.end27:                                         ; preds = %if.end24, %if.then25
  %.741 = phi i8* [ %103, %if.then25 ], [ %.640, %if.end24 ]
  %.7 = phi i8* [ %106, %if.then25 ], [ %.6, %if.end24 ]
  %107 = icmp sgt i64 %., 7
  br i1 %107, label %if.then28, label %if.end31

if.then28:                                        ; preds = %if.end27
  %108 = bitcast i8* %.741 to i32*
  %109 = load i32* %108
  %110 = getelementptr i8* %.741, i64 %1
  %111 = add i32 %109, 2
  %112 = bitcast i8* %.7 to i32*
  store i32 %111, i32* %112, !nontemporal !0
  %113 = getelementptr i8* %.7, i64 %3
  br label %if.end31

if.end31:                                         ; preds = %if.else6, %if.then28, %if.end27, %if.then5
  %.842 = phi i8* [ %33, %if.then5 ], [ %110, %if.then28 ], [ %.741, %if.end27 ], [ %.03458, %if.else6 ]
  %.8 = phi i8* [ %57, %if.then5 ], [ %113, %if.then28 ], [ %.7, %if.end27 ], [ %.03359, %if.else6 ]
  %114 = add i64 %.03260, 8
  %lsr.iv.next = add i64 %lsr.iv, -8
  %115 = icmp slt i64 %114, %0
  br i1 %115, label %loop.body, label %loop.end
}
